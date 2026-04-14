"""
ScriptedSortingPolicy — Hardcoded policy for sorting_packages task.

6-phase state machine:
  Phase 1 (APPROACH):       Rotate waist to table, move EEF above target carton
  Phase 2 (GRASP):          Lower, close gripper, lift carton
  Phase 3 (MOVE_TO_SCANNER): Rotate to scanner, place carton (barcode up)
  Phase 4 (REGRASP):        Re-pick carton from scanner
  Phase 5 (MOVE_TO_BIN):    Rotate to bin, drop carton
  Phase 6 (RETURN):         Rotate waist back to initial

All target positions (carton, scanner, bin) are queried at runtime.
IKFKSolver converts EEF targets (arm_base_link frame) → joint angles.
"""

import numpy as np
from collections import deque

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None

try:
    from geniesim.benchmark.policy.base import BasePolicy
except ImportError:
    class BasePolicy:
        def __init__(self, task_name="", sub_task_name=""):
            self.task_name = task_name
            self.sub_task_name = sub_task_name
            self.action_buffer = deque()
        def reset(self): pass
        def need_infer(self): return len(self.action_buffer) == 0
        def act(self, observations, **kwargs): pass
        def set_data_courier(self, dc): self.data_courier = dc


class ScriptedSortingPolicy(BasePolicy):
    """Hardcoded state-machine policy for sorting_packages."""

    # ── Robot constants ──────────────────────────────────────────────
    ROBOT_BASE = np.array([0.24469, 0.09325, 0.0])

    # Body joints in FK/URDF order: [bj1, bj2, bj3, bj4, bj5]
    # G2_STATES_4 body_state is [bj5,bj4,bj3,bj2,bj1] = [1.57, 0.0, -0.319, 1.344, -1.045]
    INIT_BODY = np.array([-1.04545222194, 1.34390352404, -0.31939525311, 0.0, 1.57])
    INIT_LEFT_ARM = np.array([
        0.739033, -0.717023, -1.524419, -1.537612, 0.27811, -0.925845, -0.839257,
    ])
    INIT_RIGHT_ARM = np.array([
        -0.739033, -0.717023, 1.524419, -1.537612, -0.27811, -0.925845, 0.839257,
    ])

    RIGHT_ARM_LOWER = np.array([-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -3.1067])
    RIGHT_ARM_UPPER = np.array([3.1067, 2.0944, 3.1067, 1.0472, 3.1067, 1.0472, 3.1067])

    # ── IK parameters ────────────────────────────────────────────────
    JOINT_SMOOTH_ALPHA = 0.7
    MAX_JOINT_DELTA = 0.15
    IK_POS_TOLERANCE = 0.12

    # ── Body FK chain (URDF) ─────────────────────────────────────────
    BODY_JOINTS_URDF = [
        ([0.102, 0, 0.144],      'y',  1),   # bj1 pitch
        ([-0.32627, 0, 0.15214], 'y',  1),   # bj2 pitch
        ([0.22875, -0.0019, 0],  'x',  1),   # bj3 roll
        ([0.17625, 0.0019, 0],   'y', -1),   # bj4 pitch
        ([0.0002112, -0.0024, 0.14475], 'z', 1),  # bj5 yaw
    ]
    ARM_BASE_ORIGIN = np.array([0.0, 0.0, 0.3085])
    ARM_BASE_RPY = np.array([-np.pi / 2, 0, 0])

    # ── Motion parameters ────────────────────────────────────────────
    BJ5_SPEED = 0.04          # rad/step for waist rotation
    EEF_STEP_FAST = 0.018     # m/step for fast moves
    EEF_STEP_SLOW = 0.008     # m/step for precise moves
    APPROACH_HEIGHT = 0.15    # m above target for pre-approach
    GRASP_HEIGHT = 0.02       # m above target for grasping
    LIFT_HEIGHT = 0.22        # m above target after grasping
    GRASP_HOLD_STEPS = 35     # steps to hold gripper closed
    RELEASE_HOLD_STEPS = 30   # steps to hold gripper open

    def __init__(self, task_name: str = "", sub_task_name: str = "") -> None:
        super().__init__(task_name=task_name, sub_task_name=sub_task_name)

        # IK/FK solver
        self.ikfk_solver = None
        try:
            from geniesim.utils.ikfk_utils import IKFKSolver
            arm_init = np.concatenate([self.INIT_LEFT_ARM, self.INIT_RIGHT_ARM])
            self.ikfk_solver = IKFKSolver(
                arm_init.tolist(), [0, 0, 0],
                self.INIT_BODY.tolist(),
                robot_cfg="G2_omnipicker",
            )
            print("[Policy] IKFKSolver initialized")
        except Exception as e:
            print(f"[Policy] IKFKSolver not available: {e}")

        # Scene positions — set at runtime via set_scene_positions()
        self._carton_pos = None   # target carton world position
        self._scanner_pos = None  # scanner world position
        self._bin_pos = None      # blue bin world position

        # bj5 angles — computed at runtime from scene positions
        self._bj5_table = None
        self._bj5_scanner = None
        self._bj5_bin = None

        self._init_eef_rpy = None
        self.reset()

    # ── Runtime configuration ────────────────────────────────────────

    def set_scene_positions(
        self,
        carton_pos: list[float],
        scanner_pos: list[float],
        bin_pos: list[float],
    ) -> None:
        """Set all scene target positions (world frame). Called by benchmark runner."""
        self._carton_pos = np.array(carton_pos)
        self._scanner_pos = np.array(scanner_pos)
        self._bin_pos = np.array(bin_pos)

        # Compute optimal bj5 for each target
        self._bj5_table = self._compute_bj5_for_target(self._carton_pos)
        self._bj5_scanner = self._compute_bj5_for_target(self._scanner_pos)
        self._bj5_bin = self._compute_bj5_for_target(self._bin_pos)

        print(f"[Policy] Scene positions set:")
        print(f"  carton:  {self._carton_pos} → bj5={self._bj5_table:.3f}")
        print(f"  scanner: {self._scanner_pos} → bj5={self._bj5_scanner:.3f}")
        print(f"  bin:     {self._bin_pos} → bj5={self._bj5_bin:.3f}")

    def set_carton_positions(self, positions: dict) -> None:
        """Legacy interface: accept carton dict from benchmark runner."""
        self._runtime_carton_positions = positions
        print(f"[Policy] Received {len(positions)} carton positions")
        for name, pos in positions.items():
            print(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.step_count = 0
        self.phase = "INIT"
        self.sub_step = 0         # sub-step within current phase
        self.right_grip = 0.0     # 0=open, 1=closed
        self.last_right_arm = self.INIT_RIGHT_ARM.copy()
        self._init_eef_rpy = None
        self._runtime_carton_positions = {}
        self._logged_init = False

    # ── Rotation / FK helpers ────────────────────────────────────────

    @staticmethod
    def _rot(axis: str, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        if axis == 'x':
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def body_fk(self, body_joints: np.ndarray) -> np.ndarray:
        """4x4 transform: robot base_link → arm_base_link."""
        rot = np.eye(3)
        pos = np.zeros(3)
        for i, (origin, axis, sign) in enumerate(self.BODY_JOINTS_URDF):
            pos += rot @ np.array(origin)
            rot = rot @ self._rot(axis, sign * body_joints[i])
        pos += rot @ self.ARM_BASE_ORIGIN
        rot = rot @ self._rot('x', self.ARM_BASE_RPY[0])
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    def _world_to_arm_T(self, body_joints: np.ndarray) -> np.ndarray:
        """4x4 transform: world → arm_base_link."""
        T = np.eye(4)
        T[:3, 3] = self.ROBOT_BASE
        return T @ self.body_fk(body_joints)

    def world_to_arm_base(self, world_pos: np.ndarray, body_joints: np.ndarray) -> np.ndarray:
        T = np.linalg.inv(self._world_to_arm_T(body_joints))
        return (T @ np.append(world_pos, 1.0))[:3]

    def arm_base_to_world(self, local_pos: np.ndarray, body_joints: np.ndarray) -> np.ndarray:
        T = self._world_to_arm_T(body_joints)
        return (T @ np.append(local_pos, 1.0))[:3]

    # ── bj5 computation ──────────────────────────────────────────────

    def _compute_bj5_for_target(self, target_world: np.ndarray) -> float:
        """Find bj5 angle that places the target within right-arm reach.

        Sweeps bj5 candidates to find the one where the target is closest
        to the right arm's reachable zone in the arm_base_link frame.
        """
        best_bj5 = self.INIT_BODY[4]
        best_score = float('inf')
        body = self.INIT_BODY.copy()

        for bj5 in np.linspace(-3.1, 3.1, 120):
            body[4] = bj5
            local = self.world_to_arm_base(target_world, body)
            # Right arm reaches in roughly +x direction in arm_base_link
            # Penalize lateral offset (y) and prefer targets in front (x>0)
            if local[0] < 0:
                continue
            horiz_dist = np.sqrt(local[0] ** 2 + local[1] ** 2)
            score = abs(local[1]) + 0.3 * abs(horiz_dist - 0.35)
            if score < best_score:
                best_score = score
                best_bj5 = bj5

        return best_bj5

    # ── IK helpers ───────────────────────────────────────────────────

    def _get_eef_rpy(self, quat_wxyz: np.ndarray) -> np.ndarray:
        if R is not None:
            qw, qx, qy, qz = quat_wxyz
            return R.from_quat([qx, qy, qz, qw]).as_euler('xyz')
        return np.array([0.0, 0.0, 0.0])

    def _compute_right_ik(
        self,
        target_pos: np.ndarray,
        target_rpy: np.ndarray,
        current_arm_14: np.ndarray,
        n_iter: int = 10,
    ) -> np.ndarray | None:
        """Solve IK for right arm. Returns 14-element joint array or None."""
        if self.ikfk_solver is None:
            return None
        try:
            # Keep left arm FK as-is
            left_eef = self.ikfk_solver.compute_eef(current_arm_14)["left"]
            lp = left_eef[:3]
            if R is not None:
                lq = left_eef[3:7]
                lr = R.from_quat([lq[1], lq[2], lq[3], lq[0]]).as_euler('xyz')
            else:
                lr = [0, 0, 0]

            eef_action = np.concatenate([lp, lr, target_pos, target_rpy, [0.0, 0.0]])
            result = self.ikfk_solver.eef_actions_to_joint(
                [eef_action] * n_iter, current_arm_14, [0, 0],
            )
            joints = np.array(result[-1][:14])
            joints[7:14] = np.clip(joints[7:14], self.RIGHT_ARM_LOWER, self.RIGHT_ARM_UPPER)
            return joints
        except Exception as e:
            print(f"[IK] Error: {e}")
            return None

    def _move_right_toward(
        self,
        target_local: np.ndarray,
        current_eef_pos: np.ndarray,
        current_eef_quat: np.ndarray,
        current_arm_14: np.ndarray,
        step_size: float = 0.012,
        target_rpy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Move right EEF toward target in arm_base_link frame.

        Returns (right_arm_joints_7, remaining_distance).
        """
        error = target_local - current_eef_pos
        dist = np.linalg.norm(error)
        if dist < 0.002:
            return self.last_right_arm, dist

        direction = error / dist * min(dist, step_size)
        next_pos = current_eef_pos + direction

        if target_rpy is None:
            target_rpy = (
                self._init_eef_rpy.copy()
                if self._init_eef_rpy is not None
                else self._get_eef_rpy(current_eef_quat)
            )

        result = self._compute_right_ik(next_pos, target_rpy, current_arm_14)
        if result is None:
            return self.last_right_arm, dist

        ik_right = result[7:14]

        # Convergence check
        ik_arm_14 = np.concatenate([current_arm_14[:7], ik_right])
        ik_eef = self.ikfk_solver.compute_eef(ik_arm_14)
        ik_err = np.linalg.norm(np.array(ik_eef["right"][:3]) - next_pos)
        if ik_err > self.IK_POS_TOLERANCE:
            if self.step_count % 30 == 0:
                print(f"[IK] Poor convergence: err={ik_err:.4f}, holding")
            return self.last_right_arm, dist

        # Smooth joint transition
        current_right = current_arm_14[7:14]
        delta = np.clip(ik_right - current_right, -self.MAX_JOINT_DELTA, self.MAX_JOINT_DELTA)
        smoothed = np.clip(
            current_right + self.JOINT_SMOOTH_ALPHA * delta,
            self.RIGHT_ARM_LOWER, self.RIGHT_ARM_UPPER,
        )
        self.last_right_arm = smoothed.copy()
        return smoothed, dist

    # ── Waist interpolation ──────────────────────────────────────────

    @staticmethod
    def _smooth_bj5(current: float, target: float, max_speed: float = 0.04) -> float:
        diff = target - current
        return current + np.clip(diff, -max_speed, max_speed)

    # ── Action builder ───────────────────────────────────────────────

    def _build_action(
        self,
        left_arm: np.ndarray,
        right_arm: np.ndarray,
        bj5: float,
        grip: float | None = None,
    ) -> np.ndarray:
        action = np.zeros(21)
        action[:7] = left_arm
        action[7:14] = right_arm
        action[14] = 0.0             # left gripper always open
        action[15] = grip if grip is not None else self.right_grip
        action[20] = bj5
        return action

    # ── Observation parsing ──────────────────────────────────────────

    def _parse_obs(self, observations: dict) -> dict:
        states = observations["states"]
        eef = observations["eef"]
        return {
            "left_arm": np.array(states[:7]),
            "right_arm": np.array(states[7:14]),
            "arm_14": np.array(states[:14]),
            "grip_r": states[15],
            "bj5": states[20] if len(states) > 20 else self.INIT_BODY[4],
            "body": self._get_body_joints(states),
            "r_eef_pos": np.array(eef["right"][:3]),
            "r_eef_quat": np.array(eef["right"][3:7]),
        }

    def _get_body_joints(self, states: list) -> np.ndarray:
        body = self.INIT_BODY.copy()
        for i in range(5):
            idx = 16 + i
            if len(states) > idx:
                body[i] = states[idx]
        return body

    # ── Phase management ─────────────────────────────────────────────

    def _set_phase(self, name: str) -> None:
        print(f"[Policy] {self.phase} → {name} "
              f"(step={self.step_count}, sub={self.sub_step})")
        self.phase = name
        self.sub_step = 0

    # ── Debug logging ────────────────────────────────────────────────

    def _log(self, obs: dict, target_world: np.ndarray | None = None, label: str = "") -> None:
        if self.step_count % 30 != 0:
            return
        eef_w = self.arm_base_to_world(obs["r_eef_pos"], obs["body"])
        msg = (f"[s={self.step_count}] {self.phase}:{self.sub_step} "
               f"eef_w=[{eef_w[0]:.3f},{eef_w[1]:.3f},{eef_w[2]:.3f}]")
        if target_world is not None:
            tgt_l = self.world_to_arm_base(target_world, obs["body"])
            d = np.linalg.norm(tgt_l - obs["r_eef_pos"])
            msg += f" tgt=[{target_world[0]:.3f},{target_world[1]:.3f},{target_world[2]:.3f}] d={d:.3f}"
        if label:
            msg += f" [{label}]"
        print(msg)

    # ── Initial calibration (once) ───────────────────────────────────

    def _log_init(self, obs: dict) -> None:
        if self._logged_init:
            return
        self._logged_init = True
        self._init_eef_rpy = self._get_eef_rpy(obs["r_eef_quat"])
        eef_w = self.arm_base_to_world(obs["r_eef_pos"], obs["body"])
        print("=" * 60)
        print("[Init] Calibration at step 1:")
        print(f"  body_joints: {obs['body']}")
        print(f"  right_arm:   {obs['right_arm']}")
        print(f"  EEF local:   {obs['r_eef_pos']}")
        print(f"  EEF world:   {eef_w}")
        print(f"  EEF RPY:     {self._init_eef_rpy}")
        print(f"  bj5:         {obs['bj5']:.4f}")
        if self._carton_pos is not None:
            print(f"  carton:      {self._carton_pos}")
        if self._scanner_pos is not None:
            print(f"  scanner:     {self._scanner_pos}")
        if self._bin_pos is not None:
            print(f"  bin:         {self._bin_pos}")
        # FK verification: compare body_fk output with IKFKSolver
        if self.ikfk_solver is not None:
            fk_eef = self.ikfk_solver.compute_eef(obs["arm_14"])
            print(f"  IKFKSolver EEF(R) local: {fk_eef['right'][:3]}")
            print(f"  body_fk EEF world:       {eef_w}")
            init_arm_base = self.body_fk(self.INIT_BODY)[:3, 3] + self.ROBOT_BASE
            print(f"  body_fk arm_base (INIT): {init_arm_base}")
            obs_arm_base = self.body_fk(obs["body"])[:3, 3] + self.ROBOT_BASE
            print(f"  body_fk arm_base (obs):  {obs_arm_base}")
        print("=" * 60)

    # ── Detect target carton from instruction ────────────────────────

    def _resolve_carton_from_instruction(self, instruction: str) -> None:
        """If carton_pos not yet set, try to pick from runtime positions."""
        if self._carton_pos is not None:
            return
        if not self._runtime_carton_positions:
            print("[Policy] WARNING: no carton positions available!")
            return

        # Match instruction keyword to carton type
        lower = instruction.lower()
        target_types = []
        if "yellow" in lower or "blue" in lower:
            target_types = ["028"]
        elif "black" in lower:
            target_types = ["020"]

        for name, pos in self._runtime_carton_positions.items():
            for t in target_types:
                if t in name:
                    self._carton_pos = np.array(pos)
                    print(f"[Policy] Target carton: {name} at {pos}")
                    return

        # Fallback: first carton
        name, pos = next(iter(self._runtime_carton_positions.items()))
        self._carton_pos = np.array(pos)
        print(f"[Policy] Fallback carton: {name} at {pos}")

    # ════════════════════════════════════════════════════════════════
    #  MAIN ACT METHOD
    # ════════════════════════════════════════════════════════════════

    def act(self, observations: dict, **kwargs) -> np.ndarray:
        self.step_count += 1
        self.sub_step += 1

        obs = self._parse_obs(observations)
        self._log_init(obs)

        # Resolve carton from instruction on first call
        instruction = kwargs.get("task_instruction", "")
        if self.step_count == 1 and instruction:
            self._resolve_carton_from_instruction(instruction)
            # No bj5 rotation needed for approach — carton is in front

        # Fallback positions if not set
        if self._carton_pos is None:
            self._carton_pos = np.array([0.017, 0.613, 1.19])
        if self._scanner_pos is None:
            self._scanner_pos = np.array([0.929, 0.0, 1.163])
        if self._bin_pos is None:
            self._bin_pos = np.array([0.300, -0.917, 0.837])
        if self._bj5_table is None:
            self._bj5_table = self._compute_bj5_for_target(self._carton_pos)
        if self._bj5_scanner is None:
            self._bj5_scanner = self._compute_bj5_for_target(self._scanner_pos)
        if self._bj5_bin is None:
            self._bj5_bin = self._compute_bj5_for_target(self._bin_pos)

        # Dispatch to phase handler
        handler = {
            "INIT": self._phase_init,
            "APPROACH": self._phase_approach,
            "GRASP": self._phase_grasp,
            "MOVE_TO_SCANNER": self._phase_move_to_scanner,
            "REGRASP": self._phase_regrasp,
            "MOVE_TO_BIN": self._phase_move_to_bin,
            "RETURN": self._phase_return,
            "DONE": self._phase_done,
        }.get(self.phase, self._phase_done)

        return handler(obs)

    # ════════════════════════════════════════════════════════════════
    #  PHASE 0: INIT — hold position for stabilization
    # ════════════════════════════════════════════════════════════════

    def _phase_init(self, obs: dict) -> np.ndarray:
        action = self._build_action(obs["left_arm"], self.INIT_RIGHT_ARM, obs["bj5"])
        if self.sub_step >= 30:
            self._set_phase("APPROACH")
        return action

    # ════════════════════════════════════════════════════════════════
    #  PHASE 1: APPROACH — move EEF above carton (no waist rotation)
    #    Eval trigger: Follow (gripper enters carton bbox)
    #    The carton is already in front of the robot at init pose.
    # ════════════════════════════════════════════════════════════════

    def _phase_approach(self, obs: dict) -> np.ndarray:
        # Keep current bj5 — carton is already in front, no rotation needed
        bj5_hold = obs["bj5"]

        # Move EEF above carton
        target_w = self._carton_pos.copy()
        target_w[2] += self.APPROACH_HEIGHT
        target_l = self.world_to_arm_base(target_w, obs["body"])
        new_joints, dist = self._move_right_toward(
            target_l, obs["r_eef_pos"], obs["r_eef_quat"],
            obs["arm_14"], step_size=self.EEF_STEP_FAST,
        )
        action = self._build_action(obs["left_arm"], new_joints, bj5_hold)
        self._log(obs, target_w, "above_carton")

        if dist < 0.04 or self.sub_step > 500:
            self._set_phase("GRASP")
        return action

    # ════════════════════════════════════════════════════════════════
    #  PHASE 2: GRASP — lower → close → lift
    #    Eval trigger: PickUpOnGripper (carton z > initial + 0.02)
    # ════════════════════════════════════════════════════════════════

    def _phase_grasp(self, obs: dict) -> np.ndarray:
        bj5_hold = obs["bj5"]  # keep current waist angle

        # Sub 1: lower to carton
        if self.right_grip < 0.5:
            target_w = self._carton_pos.copy()
            target_w[2] += self.GRASP_HEIGHT
            target_l = self.world_to_arm_base(target_w, obs["body"])
            new_joints, dist = self._move_right_toward(
                target_l, obs["r_eef_pos"], obs["r_eef_quat"],
                obs["arm_14"], step_size=self.EEF_STEP_SLOW,
            )
            action = self._build_action(obs["left_arm"], new_joints, bj5_hold)
            self._log(obs, target_w, "lowering")

            if dist < 0.03 or self.sub_step > 300:
                # Close gripper
                self.right_grip = 1.0
                self.sub_step = 0  # reset for hold counting
                print(f"[Policy] Gripper closing at step {self.step_count}")
            return action

        # Sub 2: hold gripper closed
        if self.sub_step <= self.GRASP_HOLD_STEPS:
            return self._build_action(
                obs["left_arm"], self.last_right_arm, bj5_hold, grip=1.0,
            )

        # Sub 3: lift carton
        target_w = self._carton_pos.copy()
        target_w[2] += self.LIFT_HEIGHT
        target_l = self.world_to_arm_base(target_w, obs["body"])
        new_joints, dist = self._move_right_toward(
            target_l, obs["r_eef_pos"], obs["r_eef_quat"],
            obs["arm_14"], step_size=self.EEF_STEP_FAST,
        )
        action = self._build_action(obs["left_arm"], new_joints, bj5_hold, grip=1.0)
        self._log(obs, target_w, "lifting")

        if dist < 0.04 or self.sub_step > self.GRASP_HOLD_STEPS + 200:
            self._set_phase("MOVE_TO_SCANNER")
        return action

    # ════════════════════════════════════════════════════════════════
    #  PHASE 3: MOVE_TO_SCANNER — retract → rotate → lower → release
    #    Eval trigger: Inside (carton in scanner AABB) + Upright (<5°)
    # ════════════════════════════════════════════════════════════════

    def _phase_move_to_scanner(self, obs: dict) -> np.ndarray:
        bj5 = obs["bj5"]

        # Sub 1: rotate waist to scanner
        if abs(bj5 - self._bj5_scanner) > 0.05 and self.sub_step < 200:
            new_bj5 = self._smooth_bj5(bj5, self._bj5_scanner, self.BJ5_SPEED)
            action = self._build_action(
                obs["left_arm"], self.last_right_arm, new_bj5, grip=1.0,
            )
            self._log(obs, self._scanner_pos, "rotating_to_scanner")
            return action

        # Sub 2: move above scanner
        target_w = self._scanner_pos.copy()
        target_w[2] += self.APPROACH_HEIGHT
        target_l = self.world_to_arm_base(target_w, obs["body"])
        new_joints, dist_above = self._move_right_toward(
            target_l, obs["r_eef_pos"], obs["r_eef_quat"],
            obs["arm_14"], step_size=self.EEF_STEP_FAST,
        )

        if dist_above > 0.05 and self.sub_step < 500:
            action = self._build_action(
                obs["left_arm"], new_joints, self._bj5_scanner, grip=1.0,
            )
            self._log(obs, target_w, "above_scanner")
            return action

        # Sub 3: lower onto scanner
        target_w = self._scanner_pos.copy()
        target_w[2] += self.GRASP_HEIGHT
        target_l = self.world_to_arm_base(target_w, obs["body"])
        new_joints, dist = self._move_right_toward(
            target_l, obs["r_eef_pos"], obs["r_eef_quat"],
            obs["arm_14"], step_size=self.EEF_STEP_SLOW,
        )

        if dist > 0.03 and self.sub_step < 700:
            action = self._build_action(
                obs["left_arm"], new_joints, self._bj5_scanner, grip=1.0,
            )
            self._log(obs, target_w, "lowering_scanner")
            return action

        # Sub 4: release
        self.right_grip = 0.0
        action = self._build_action(
            obs["left_arm"], self.last_right_arm, self._bj5_scanner, grip=0.0,
        )
        if self.sub_step > 700 + self.RELEASE_HOLD_STEPS:
            self._set_phase("REGRASP")
        return action

    # ════════════════════════════════════════════════════════════════
    #  PHASE 4: REGRASP — lift hand → wait → lower → close → lift
    #    Eval trigger: PickUpOnGripper (carton z > scanner + 0.02)
    # ════════════════════════════════════════════════════════════════

    def _phase_regrasp(self, obs: dict) -> np.ndarray:
        bj5_hold = self._bj5_scanner

        # Sub 1: lift hand above scanner (clear the carton)
        if self.sub_step <= 80:
            target_w = self._scanner_pos.copy()
            target_w[2] += self.APPROACH_HEIGHT
            target_l = self.world_to_arm_base(target_w, obs["body"])
            new_joints, _ = self._move_right_toward(
                target_l, obs["r_eef_pos"], obs["r_eef_quat"],
                obs["arm_14"], step_size=self.EEF_STEP_FAST,
            )
            return self._build_action(obs["left_arm"], new_joints, bj5_hold, grip=0.0)

        # Sub 2: wait for carton to settle
        if self.sub_step <= 140:
            return self._build_action(
                obs["left_arm"], self.last_right_arm, bj5_hold, grip=0.0,
            )

        # Sub 3: lower to carton on scanner
        if self.right_grip < 0.5:
            target_w = self._scanner_pos.copy()
            target_w[2] += self.GRASP_HEIGHT
            target_l = self.world_to_arm_base(target_w, obs["body"])
            new_joints, dist = self._move_right_toward(
                target_l, obs["r_eef_pos"], obs["r_eef_quat"],
                obs["arm_14"], step_size=self.EEF_STEP_SLOW,
            )
            action = self._build_action(obs["left_arm"], new_joints, bj5_hold, grip=0.0)
            self._log(obs, target_w, "regrasp_lower")

            if dist < 0.03 or self.sub_step > 350:
                self.right_grip = 1.0
                self.sub_step = 350  # sync for hold counting
                print(f"[Policy] Re-grasp closing at step {self.step_count}")
            return action

        # Sub 4: hold
        if self.sub_step <= 350 + self.GRASP_HOLD_STEPS:
            return self._build_action(
                obs["left_arm"], self.last_right_arm, bj5_hold, grip=1.0,
            )

        # Sub 5: lift from scanner
        target_w = self._scanner_pos.copy()
        target_w[2] += self.LIFT_HEIGHT
        target_l = self.world_to_arm_base(target_w, obs["body"])
        new_joints, dist = self._move_right_toward(
            target_l, obs["r_eef_pos"], obs["r_eef_quat"],
            obs["arm_14"], step_size=self.EEF_STEP_FAST,
        )
        action = self._build_action(obs["left_arm"], new_joints, bj5_hold, grip=1.0)
        self._log(obs, target_w, "regrasp_lift")

        if dist < 0.05 or self.sub_step > 350 + self.GRASP_HOLD_STEPS + 200:
            self._set_phase("MOVE_TO_BIN")
        return action

    # ════════════════════════════════════════════════════════════════
    #  PHASE 5: MOVE_TO_BIN — rotate → extend → release
    #    Eval trigger: Inside (carton in bin AABB)
    # ════════════════════════════════════════════════════════════════

    def _phase_move_to_bin(self, obs: dict) -> np.ndarray:
        bj5 = obs["bj5"]

        # Sub 1: rotate waist to bin
        if abs(bj5 - self._bj5_bin) > 0.05 and self.sub_step < 150:
            new_bj5 = self._smooth_bj5(bj5, self._bj5_bin, self.BJ5_SPEED)
            action = self._build_action(
                obs["left_arm"], self.last_right_arm, new_bj5, grip=1.0,
            )
            self._log(obs, self._bin_pos, "rotating_to_bin")
            return action

        # Sub 2: extend arm toward bin (may not fully reach, that's OK)
        arm_w = self.arm_base_to_world(np.zeros(3), obs["body"])
        to_bin = self._bin_pos - arm_w
        to_bin_dir = to_bin / (np.linalg.norm(to_bin) + 1e-8)
        reach_w = arm_w + to_bin_dir * 0.65
        reach_w[2] = max(self._bin_pos[2] + 0.15, reach_w[2])
        target_l = self.world_to_arm_base(reach_w, obs["body"])

        new_joints, dist = self._move_right_toward(
            target_l, obs["r_eef_pos"], obs["r_eef_quat"],
            obs["arm_14"], step_size=self.EEF_STEP_FAST,
        )

        if dist > 0.06 and self.sub_step < 400:
            action = self._build_action(
                obs["left_arm"], new_joints, self._bj5_bin, grip=1.0,
            )
            self._log(obs, reach_w, "extending_to_bin")
            return action

        # Sub 3: release
        self.right_grip = 0.0
        action = self._build_action(
            obs["left_arm"], self.last_right_arm, self._bj5_bin, grip=0.0,
        )
        if self.sub_step > 400 + self.RELEASE_HOLD_STEPS:
            self._set_phase("RETURN")
        return action

    # ════════════════════════════════════════════════════════════════
    #  PHASE 6: RETURN — rotate waist back to initial
    # ════════════════════════════════════════════════════════════════

    def _phase_return(self, obs: dict) -> np.ndarray:
        bj5 = obs["bj5"]
        init_bj5 = self.INIT_BODY[4]
        new_bj5 = self._smooth_bj5(bj5, init_bj5, self.BJ5_SPEED)
        action = self._build_action(
            obs["left_arm"], self.last_right_arm, new_bj5, grip=0.0,
        )
        if abs(bj5 - init_bj5) < 0.1 or self.sub_step > 150:
            self._set_phase("DONE")
        return action

    # ── DONE ─────────────────────────────────────────────────────────

    def _phase_done(self, obs: dict) -> np.ndarray:
        return self._build_action(
            obs["left_arm"], self.last_right_arm, self.INIT_BODY[4], grip=0.0,
        )


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    policy = ScriptedSortingPolicy()

    print("=== Body FK at init ===")
    T = policy.body_fk(policy.INIT_BODY)
    arm_w = T[:3, 3] + policy.ROBOT_BASE
    print(f"arm_base_link world: {arm_w}")

    print("\n=== bj5 sweep for targets ===")
    targets = {
        "carton": np.array([0.017, 0.613, 1.19]),
        "scanner": np.array([0.929, 0.0, 1.163]),
        "bin": np.array([0.300, -0.917, 0.837]),
    }
    for name, pos in targets.items():
        bj5 = policy._compute_bj5_for_target(pos)
        body = policy.INIT_BODY.copy()
        body[4] = bj5
        local = policy.world_to_arm_base(pos, body)
        print(f"  {name}: bj5={bj5:.3f} local={local} dist={np.linalg.norm(local):.3f}")
