"""
ScriptedSortingPolicy - Hardcoded policy for sorting_packages task.

State machine policy that controls the G2 robot to:
1. Pick up a target carton from the table (right arm)
2. Place it on the scanner (barcode up)
3. Re-pick from scanner
4. Place in the bin (drop strategy since bin is beyond arm reach)

Uses body FK for coordinate transforms and IK solver for arm control.

Set DIAG_MODE = True to run diagnostic hold-position tests first.
"""

import numpy as np
from collections import deque

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None

# Try importing benchmark base
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

    # ── Diagnostic mode ─────────────────────────────────────────────
    # When True: runs hold-position tests before normal operation.
    # Set to False once diagnostics confirm arm holds correctly.
    DIAG_MODE = False
    DIAG_HOLD_STEPS = 100    # hold INIT_RIGHT_ARM constant
    DIAG_ECHO_STEPS = 100    # echo observed joints back
    DIAG_WAIST_STEPS = 200   # rotate waist while holding arm

    # ── Robot constants ──────────────────────────────────────────────
    ROBOT_BASE = np.array([0.24469, 0.09325, 0.0])

    # G2_STATES_4 initial values
    INIT_BODY = np.array([1.57, 0.0, -0.31939525311, 1.34390352404, -1.04545222194])
    INIT_LEFT_ARM = np.array([0.739033, -0.717023, -1.524419, -1.537612, 0.27811, -0.925845, -0.839257])
    INIT_RIGHT_ARM = np.array([-0.739033, -0.717023, 1.524419, -1.537612, -0.27811, -0.925845, 0.839257])

    # Right arm joint limits (from URDF)
    RIGHT_ARM_LOWER = np.array([-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -3.1067])
    RIGHT_ARM_UPPER = np.array([3.1067, 2.0944, 3.1067, 1.0472, 3.1067, 1.0472, 3.1067])

    # Gripper offset: distance from arm_r_end_link (FK point) to gripper center
    GRIPPER_Z_OFFSET = 0.143

    # ── IK control parameters ────────────────────────────────────────
    JOINT_SMOOTH_ALPHA = 0.5   # blend factor: 0=keep current, 1=full IK
    MAX_JOINT_DELTA = 0.10     # max joint change per step (rad)
    IK_POS_TOLERANCE = 0.08    # max acceptable IK position error (m)

    # ── Scene positions (world frame) ────────────────────────────────
    SCANNER_POS = np.array([0.929, 0.0, 1.163])
    BIN_POS = np.array([0.300, -0.917, 0.837])

    # Default carton world positions (will be overridden by runtime query)
    CARTON_POSITIONS = {
        0: np.array([0.017, 0.613, 1.19]),     # blue/yellow package (scene_info)
        1: np.array([0.467, 0.59, 1.158]),      # black package
    }

    # Map instruction keywords to variant
    INSTRUCTION_TO_VARIANT = {
        "yellow": 0,
        "black": 1,
        "blue": 0,
    }

    # ── Optimal body_joint5 values for facing each target ────────────
    BJ5_TABLE = 2.0        # face toward table (left side)
    BJ5_SCANNER = -1.045   # face toward scanner (forward-right)
    BJ5_BIN = -2.0         # face toward bin (right side)

    # ── URDF body chain parameters ───────────────────────────────────
    BODY_JOINTS_URDF = [
        ([0.102, 0, 0.144],      'y',  1),   # bj1 - pitch
        ([-0.32627, 0, 0.15214], 'y',  1),   # bj2 - pitch
        ([0.22875, -0.0019, 0],  'x',  1),   # bj3 - roll
        ([0.17625, 0.0019, 0],   'y', -1),   # bj4 - pitch
        ([0.0002112, -0.0024, 0.14475], 'z', 1),  # bj5 - yaw
    ]
    ARM_BASE_ORIGIN = np.array([0.0, 0.0, 0.3085])
    ARM_BASE_RPY = np.array([-np.pi / 2, 0, 0])

    def __init__(self, task_name="", sub_task_name=""):
        super().__init__(task_name=task_name, sub_task_name=sub_task_name)

        self.ikfk_solver = None
        try:
            from geniesim.utils.ikfk_utils import IKFKSolver
            arm_init = np.concatenate([self.INIT_LEFT_ARM, self.INIT_RIGHT_ARM])
            self.ikfk_solver = IKFKSolver(
                arm_init.tolist(), [0, 0, 0],
                self.INIT_BODY.tolist(),
                robot_cfg="G2_omnipicker",
            )
            print("[ScriptedPolicy] IK/FK solver initialized")
        except Exception as e:
            print(f"[ScriptedPolicy] IK solver not available: {e}")

        self.variant = 0
        self._detect_variant = True
        self._init_eef_logged = False
        self._init_eef_rpy = None
        self._calibrated = False
        self._runtime_carton_positions = {}  # from environment query
        self.reset()

    # ── Accept dynamic carton positions from environment ─────────────
    def set_carton_positions(self, positions):
        """Called by the benchmark runner with actual carton world positions.

        Args:
            positions: dict of {carton_id: [x, y, z]}
        """
        self._runtime_carton_positions = positions
        print(f"[Policy] Received {len(positions)} carton positions from environment")
        for name, pos in positions.items():
            print(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    def _get_target_carton_pos(self):
        """Get the target carton world position, using runtime data if available."""
        if self._runtime_carton_positions:
            # Find closest carton matching our variant
            # For now, use the first carton with "028" (blue/yellow) for variant 0
            # or "020" for variant 1
            target_type = "028" if self.variant == 0 else "020"
            for name, pos in self._runtime_carton_positions.items():
                if target_type in name:
                    return np.array(pos)
            # Fallback: use the closest carton on the "scattered" set
            # (type_028, type_029, type_030, type_020)
            scattered_types = ["028", "029", "030", "020"]
            for name, pos in self._runtime_carton_positions.items():
                for t in scattered_types:
                    if t in name:
                        return np.array(pos)
            # Last fallback: first carton
            first_pos = next(iter(self._runtime_carton_positions.values()))
            return np.array(first_pos)

        # No runtime data, use defaults
        return self.CARTON_POSITIONS.get(self.variant, self.CARTON_POSITIONS[0]).copy()

    # ── Reset ────────────────────────────────────────────────────────
    def reset(self):
        self.step_count = 0
        self.phase = "DIAG_HOLD" if self.DIAG_MODE else "INIT"
        self.phase_step = 0
        self.right_grip = 0.0   # 0=open, 1=close
        self.last_right_arm = self.INIT_RIGHT_ARM.copy()
        self.target_bj5 = self.INIT_BODY[4]
        self._detect_variant = True
        self._init_eef_logged = False
        self._calibrated = False
        self._diag_init_obs_arm = None  # observed arm at step 1
        return None

    # ── Rotation helpers ─────────────────────────────────────────────
    @staticmethod
    def _rot(axis, angle):
        c, s = np.cos(angle), np.sin(angle)
        if axis == 'x':
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:  # z
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # ── Body forward kinematics ──────────────────────────────────────
    def body_fk(self, body_joints):
        """Compute 4x4 transform from robot base_link to arm_base_link."""
        bj = list(body_joints)
        rot = np.eye(3)
        pos = np.zeros(3)

        for i, (origin, axis, sign) in enumerate(self.BODY_JOINTS_URDF):
            pos += rot @ np.array(origin)
            rot = rot @ self._rot(axis, sign * bj[i])

        pos += rot @ self.ARM_BASE_ORIGIN
        rot = rot @ self._rot('x', self.ARM_BASE_RPY[0])

        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    def _get_world_to_arm_T(self, body_joints):
        """Get the full world-to-arm_base_link transform matrix."""
        T_base_to_arm = self.body_fk(body_joints)
        T_world = np.eye(4)
        T_world[:3, 3] = self.ROBOT_BASE
        T_world = T_world @ T_base_to_arm
        return T_world

    def world_to_arm_base(self, world_pos, body_joints):
        """Convert world position to arm_base_link frame."""
        T_world = self._get_world_to_arm_T(body_joints)
        T_inv = np.linalg.inv(T_world)
        p = T_inv @ np.append(world_pos, 1.0)
        return p[:3]

    def arm_base_to_world(self, local_pos, body_joints):
        """Convert arm_base_link position to world frame."""
        T_world = self._get_world_to_arm_T(body_joints)
        p = T_world @ np.append(local_pos, 1.0)
        return p[:3]

    # ── IK helper ────────────────────────────────────────────────────
    def _compute_right_ik(self, target_pos, target_rpy, current_arm_14, n_iter=10):
        """Compute right arm joints via IK solver with multiple iterations."""
        if self.ikfk_solver is None:
            return None
        try:
            left_eef = self.ikfk_solver.compute_eef(current_arm_14)
            lp = left_eef["left"][:3]
            lq = left_eef["left"][3:7]
            if R is not None:
                lr = R.from_quat([lq[1], lq[2], lq[3], lq[0]]).as_euler('xyz')
            else:
                lr = [0, 0, 0]
            left_xyzrpy = np.concatenate([lp, lr])

            right_xyzrpy = np.concatenate([target_pos, target_rpy])
            eef_action = np.concatenate([left_xyzrpy, right_xyzrpy, [0.0, 0.0]])

            eef_actions = [eef_action] * n_iter
            result = self.ikfk_solver.eef_actions_to_joint(
                eef_actions, current_arm_14, [0, 0]
            )
            joints_14 = np.array(result[-1][:14])

            right = joints_14[7:14]
            right = np.clip(right, self.RIGHT_ARM_LOWER, self.RIGHT_ARM_UPPER)
            joints_14[7:14] = right
            return joints_14
        except Exception as e:
            print(f"[IK] Error: {e}")
            return None

    def _get_right_eef_rpy(self, eef_quat):
        """Convert qw,qx,qy,qz to roll,pitch,yaw."""
        if R is not None:
            qw, qx, qy, qz = eef_quat
            return R.from_quat([qx, qy, qz, qw]).as_euler('xyz')
        return np.array([0, 0, 0])

    # ── Smooth interpolation ─────────────────────────────────────────
    @staticmethod
    def _smooth_bj5(current_bj5, target_bj5, max_speed=0.04):
        """Smoothly move bj5 toward target with speed limit (rad/step)."""
        diff = target_bj5 - current_bj5
        if abs(diff) > max_speed:
            diff = max_speed * np.sign(diff)
        return current_bj5 + diff

    # ── Move right arm toward EEF target ─────────────────────────────
    def _move_right_toward(self, target_pos_local, current_eef_pos, current_eef_quat,
                           current_arm_14, step_size=0.015, target_rpy=None,
                           n_iter=10):
        """Move right EEF toward target_pos_local with small steps.

        Returns (new_right_joints_7, distance_remaining).
        """
        error = target_pos_local - current_eef_pos
        dist = np.linalg.norm(error)
        if dist < 0.002:
            return self.last_right_arm, dist

        if dist > step_size:
            direction = error / dist * step_size
        else:
            direction = error
        next_pos = current_eef_pos + direction

        if target_rpy is None:
            if self._init_eef_rpy is not None:
                target_rpy = self._init_eef_rpy.copy()
            else:
                target_rpy = self._get_right_eef_rpy(current_eef_quat)

        result = self._compute_right_ik(next_pos, target_rpy, current_arm_14,
                                        n_iter=n_iter)

        if result is not None:
            ik_right = result[7:14]
            ik_right = np.clip(ik_right, self.RIGHT_ARM_LOWER, self.RIGHT_ARM_UPPER)

            # IK convergence check
            ik_arm_14 = np.concatenate([current_arm_14[:7], ik_right])
            ik_eef = self.ikfk_solver.compute_eef(ik_arm_14)
            ik_eef_pos = np.array(ik_eef["right"][:3])
            ik_error = np.linalg.norm(ik_eef_pos - next_pos)

            if ik_error > self.IK_POS_TOLERANCE:
                if self.step_count % 30 == 0:
                    print(f"[IK] Poor convergence: ik_err={ik_error:.4f} "
                          f"(tol={self.IK_POS_TOLERANCE}), holding position")
                return self.last_right_arm, dist

            # Joint-space smoothing
            current_right = current_arm_14[7:14]
            delta = ik_right - current_right
            delta = np.clip(delta, -self.MAX_JOINT_DELTA, self.MAX_JOINT_DELTA)
            smoothed = current_right + self.JOINT_SMOOTH_ALPHA * delta
            smoothed = np.clip(smoothed, self.RIGHT_ARM_LOWER, self.RIGHT_ARM_UPPER)
            self.last_right_arm = smoothed.copy()
            return smoothed, dist
        return self.last_right_arm, dist

    # ── Get current body joints from observation ─────────────────────
    def _get_body_joints(self, states):
        """Extract body joints from observation states.

        PiEnv appends waist joints via G2_WAIST_JOINT_NAMES[::-1],
        yielding order [bj1, bj2, bj3, bj4, bj5] at indices 16-20.
        """
        bj1 = states[16] if len(states) > 16 else self.INIT_BODY[0]
        bj2 = states[17] if len(states) > 17 else self.INIT_BODY[1]
        bj3 = states[18] if len(states) > 18 else self.INIT_BODY[2]
        bj4 = states[19] if len(states) > 19 else self.INIT_BODY[3]
        bj5 = states[20] if len(states) > 20 else self.INIT_BODY[4]
        return np.array([bj1, bj2, bj3, bj4, bj5])

    # ── Phase transition ─────────────────────────────────────────────
    def _next_phase(self, phase_name):
        print(f"[Policy] {self.phase} -> {phase_name} "
              f"(step {self.step_count}, phase_steps={self.phase_step})")
        self.phase = phase_name
        self.phase_step = 0

    # ── Variant detection ────────────────────────────────────────────
    def _try_detect_variant(self, kwargs):
        """Detect variant from task_instruction."""
        instruction = kwargs.get("task_instruction", "")
        if instruction:
            lower = instruction.lower()
            for keyword, var_id in self.INSTRUCTION_TO_VARIANT.items():
                if keyword in lower:
                    self.variant = var_id
                    print(f"[Policy] Detected variant {var_id} from instruction "
                          f"(keyword: '{keyword}')")
                    break
            else:
                print(f"[Policy] No variant keyword found, "
                      f"defaulting to variant {self.variant}")
        else:
            print(f"[Policy] No task_instruction, "
                  f"defaulting to variant {self.variant}")

        carton_pos = self._get_target_carton_pos()
        print(f"[Policy] Target carton world position: {carton_pos}")

    # ── Build default action ─────────────────────────────────────────
    def _default_action(self, left_arm, right_arm, bj5):
        """Build a default hold-position action."""
        action = np.zeros(21)
        action[:7] = left_arm
        action[7:14] = right_arm
        action[14] = 0.0            # left gripper open
        action[15] = self.right_grip
        action[20] = bj5
        return action

    # ── Periodic debug logging ───────────────────────────────────────
    def _log_debug(self, r_eef_pos, body_joints, target_world=None, label=""):
        """Log debug info periodically (every 30 steps)."""
        if self.step_count % 30 != 0:
            return
        eef_world = self.arm_base_to_world(r_eef_pos, body_joints)
        arm_base_w = self.arm_base_to_world(np.zeros(3), body_joints)
        msg = (f"[DBG s={self.step_count}] phase={self.phase} ps={self.phase_step} "
               f"eef_local=[{r_eef_pos[0]:.3f},{r_eef_pos[1]:.3f},{r_eef_pos[2]:.3f}] "
               f"eef_world=[{eef_world[0]:.3f},{eef_world[1]:.3f},{eef_world[2]:.3f}] "
               f"arm_base=[{arm_base_w[0]:.3f},{arm_base_w[1]:.3f},{arm_base_w[2]:.3f}]")
        if target_world is not None:
            target_local = self.world_to_arm_base(target_world, body_joints)
            msg += (f" tgt_w=[{target_world[0]:.3f},{target_world[1]:.3f},{target_world[2]:.3f}]"
                    f" tgt_l=[{target_local[0]:.3f},{target_local[1]:.3f},{target_local[2]:.3f}]"
                    f" d={np.linalg.norm(target_local - r_eef_pos):.4f}")
        if label:
            msg += f" [{label}]"
        print(msg)

    # ── Diagnostic logging (per step) ────────────────────────────────
    def _log_diag(self, phase, obs_right, cmd_right, bj5, eef_pos):
        """Log observed vs commanded arm joints every 10 steps during diagnostics."""
        if self.step_count % 10 != 0:
            return
        diff = np.array(obs_right) - np.array(cmd_right)
        max_diff = np.max(np.abs(diff))
        print(f"[DIAG s={self.step_count}] phase={phase} bj5={bj5:.4f}")
        print(f"  obs_arm: [{', '.join(f'{v:.5f}' for v in obs_right)}]")
        print(f"  cmd_arm: [{', '.join(f'{v:.5f}' for v in cmd_right)}]")
        print(f"  diff:    [{', '.join(f'{v:.5f}' for v in diff)}]  max={max_diff:.5f}")
        print(f"  eef_pos: [{eef_pos[0]:.4f}, {eef_pos[1]:.4f}, {eef_pos[2]:.4f}]")

    # ── Main act method ──────────────────────────────────────────────
    def act(self, observations, **kwargs):
        self.step_count += 1
        self.phase_step += 1

        states = observations["states"]
        eef = observations["eef"]

        # Current state
        left_arm = np.array(states[:7])
        right_arm = np.array(states[7:14])
        grip_r_state = states[15]
        bj5 = states[20] if len(states) > 20 else self.INIT_BODY[4]

        current_arm_14 = np.concatenate([left_arm, right_arm])
        r_eef_pos = np.array(eef["right"][:3])
        r_eef_quat = np.array(eef["right"][3:7])

        body_joints = self._get_body_joints(states)

        # ── Initial calibration logging (step 1) ──
        if not self._init_eef_logged:
            self._init_eef_logged = True
            self._diag_init_obs_arm = right_arm.copy()
            r_eef_world = self.arm_base_to_world(r_eef_pos, body_joints)
            arm_base_world = self.arm_base_to_world(np.zeros(3), body_joints)
            print("=" * 70)
            print("[CALIBRATION] Initial state at step 1:")
            print(f"  states length: {len(states)}")
            print(f"  body_joints (observed): {body_joints}")
            print(f"  body_joints (expected): {self.INIT_BODY}")
            print(f"  right_arm (observed): {right_arm}")
            print(f"  right_arm (expected): {self.INIT_RIGHT_ARM}")
            arm_diff = right_arm - self.INIT_RIGHT_ARM
            print(f"  right_arm diff:       {arm_diff}")
            print(f"  right_arm max diff:   {np.max(np.abs(arm_diff)):.6f}")
            print(f"  right EEF local (arm_base frame): {r_eef_pos}")
            print(f"  right EEF world (computed): {r_eef_world}")
            print(f"  right EEF quat: {r_eef_quat}")
            print(f"  arm_base_link world pos: {arm_base_world}")
            print(f"  bj5: {bj5}")
            print(f"  gripper state: {grip_r_state}")

            # Cache initial EEF orientation
            self._init_eef_rpy = self._get_right_eef_rpy(r_eef_quat)
            print(f"  init EEF RPY: {self._init_eef_rpy}")

            # Show distances to all targets
            for name, pos in [("scanner", self.SCANNER_POS),
                              ("bin", self.BIN_POS)]:
                local = self.world_to_arm_base(pos, body_joints)
                print(f"  {name}: world={pos} local={local} "
                      f"dist={np.linalg.norm(local):.3f}")

            # Show carton positions
            carton_pos = self._get_target_carton_pos()
            local = self.world_to_arm_base(carton_pos, body_joints)
            print(f"  target_carton: world={carton_pos} local={local} "
                  f"dist={np.linalg.norm(local):.3f}")

            if self._runtime_carton_positions:
                print(f"  Runtime carton positions ({len(self._runtime_carton_positions)}):")
                for name, pos in self._runtime_carton_positions.items():
                    local_c = self.world_to_arm_base(np.array(pos), body_joints)
                    print(f"    {name}: world={pos} local=[{local_c[0]:.3f},{local_c[1]:.3f},{local_c[2]:.3f}] "
                          f"dist={np.linalg.norm(local_c):.3f}")
            print("=" * 70)

        # Detect variant on first call
        if self._detect_variant:
            self._detect_variant = False
            self._try_detect_variant(kwargs)

        # Get target carton position
        carton_pos = self._get_target_carton_pos()

        # ── DIAGNOSTIC PHASES ────────────────────────────────────────
        if self.phase == "DIAG_HOLD":
            # Hold constant INIT_RIGHT_ARM — test if arm drifts
            cmd_right = self.INIT_RIGHT_ARM.copy()
            action = self._default_action(left_arm, cmd_right, bj5)
            action[20] = self.INIT_BODY[4]  # hold waist too
            self._log_diag("DIAG_HOLD", right_arm, cmd_right, bj5, r_eef_pos)
            if self.phase_step >= self.DIAG_HOLD_STEPS:
                self._next_phase("DIAG_ECHO")
            return action

        elif self.phase == "DIAG_ECHO":
            # Echo observed joints back — test if echo loop is stable
            cmd_right = right_arm.copy()
            action = self._default_action(left_arm, cmd_right, bj5)
            action[20] = self.INIT_BODY[4]
            self._log_diag("DIAG_ECHO", right_arm, cmd_right, bj5, r_eef_pos)
            if self.phase_step >= self.DIAG_ECHO_STEPS:
                self._next_phase("DIAG_WAIST")
            return action

        elif self.phase == "DIAG_WAIST":
            # Hold arm constant while rotating waist — test coupling
            cmd_right = self.INIT_RIGHT_ARM.copy()
            target_bj5 = self.BJ5_TABLE
            new_bj5 = self._smooth_bj5(bj5, target_bj5, 0.03)
            action = self._default_action(left_arm, cmd_right, new_bj5)
            self._log_diag("DIAG_WAIST", right_arm, cmd_right, bj5, r_eef_pos)

            # After waist test, log FK at new position
            if self.phase_step % 50 == 0:
                arm_base_w = self.arm_base_to_world(np.zeros(3), body_joints)
                carton_local = self.world_to_arm_base(carton_pos, body_joints)
                print(f"[DIAG_WAIST s={self.step_count}] bj5={bj5:.4f} "
                      f"arm_base_world=[{arm_base_w[0]:.3f},{arm_base_w[1]:.3f},{arm_base_w[2]:.3f}] "
                      f"carton_local=[{carton_local[0]:.3f},{carton_local[1]:.3f},{carton_local[2]:.3f}] "
                      f"carton_dist={np.linalg.norm(carton_local):.3f}")

            if self.phase_step >= self.DIAG_WAIST_STEPS:
                # Diagnostics complete — proceed to normal operation
                print("[DIAG] Diagnostics complete. Starting normal operation.")
                self._next_phase("INIT")
            return action

        # ── NORMAL OPERATION ─────────────────────────────────────────

        # Build default action (hold everything)
        action = self._default_action(left_arm, self.last_right_arm, bj5)

        if self.phase == "INIT":
            # Hold position with constant INIT joints for stability
            action[7:14] = self.INIT_RIGHT_ARM
            action[20] = bj5  # hold waist
            if self.phase_step >= 30:
                self._next_phase("ROTATE_TO_TABLE")

        elif self.phase == "ROTATE_TO_TABLE":
            # Hold arm constant while rotating waist to face table
            action[7:14] = self.INIT_RIGHT_ARM
            action[20] = self._smooth_bj5(bj5, self.BJ5_TABLE, 0.04)
            self.last_right_arm = self.INIT_RIGHT_ARM.copy()

            if self.step_count % 30 == 0:
                arm_base_w = self.arm_base_to_world(np.zeros(3), body_joints)
                carton_local = self.world_to_arm_base(carton_pos, body_joints)
                print(f"[ROTATE s={self.step_count}] bj5={bj5:.4f}->{self.BJ5_TABLE:.2f} "
                      f"arm_base=[{arm_base_w[0]:.3f},{arm_base_w[1]:.3f},{arm_base_w[2]:.3f}] "
                      f"carton_local=[{carton_local[0]:.3f},{carton_local[1]:.3f},{carton_local[2]:.3f}] "
                      f"dist={np.linalg.norm(carton_local):.3f}")

            if abs(bj5 - self.BJ5_TABLE) < 0.05 and self.phase_step > 20:
                self._next_phase("APPROACH_ABOVE_CARTON")
            elif self.phase_step > 150:
                self._next_phase("APPROACH_ABOVE_CARTON")

        elif self.phase == "APPROACH_ABOVE_CARTON":
            action[20] = self.BJ5_TABLE
            target_world = carton_pos.copy()
            target_world[2] += 0.15
            target_local = self.world_to_arm_base(target_world, body_joints)
            new_joints, dist = self._move_right_toward(
                target_local, r_eef_pos, r_eef_quat, current_arm_14,
                step_size=0.012, n_iter=15
            )
            action[7:14] = new_joints
            self._log_debug(r_eef_pos, body_joints, target_world, "APPROACH_ABOVE")
            if dist < 0.04 or self.phase_step > 400:
                self._next_phase("LOWER_TO_CARTON")

        elif self.phase == "LOWER_TO_CARTON":
            action[20] = self.BJ5_TABLE
            target_world = carton_pos.copy()
            target_world[2] += 0.02
            target_local = self.world_to_arm_base(target_world, body_joints)
            new_joints, dist = self._move_right_toward(
                target_local, r_eef_pos, r_eef_quat, current_arm_14,
                step_size=0.008, n_iter=15
            )
            action[7:14] = new_joints
            self._log_debug(r_eef_pos, body_joints, target_world, "LOWER_TO")
            if dist < 0.03 or self.phase_step > 250:
                self._next_phase("GRASP_CARTON")

        elif self.phase == "GRASP_CARTON":
            action[20] = self.BJ5_TABLE
            action[7:14] = self.last_right_arm
            self.right_grip = 1.0
            action[15] = 1.0
            if self.phase_step >= 35:
                self._next_phase("LIFT_CARTON")

        elif self.phase == "LIFT_CARTON":
            action[20] = self.BJ5_TABLE
            action[15] = 1.0
            target_world = carton_pos.copy()
            target_world[2] += 0.22
            target_local = self.world_to_arm_base(target_world, body_joints)
            new_joints, dist = self._move_right_toward(
                target_local, r_eef_pos, r_eef_quat, current_arm_14,
                step_size=0.012, n_iter=15
            )
            action[7:14] = new_joints
            self._log_debug(r_eef_pos, body_joints, target_world, "LIFT")
            if dist < 0.04 or self.phase_step > 150:
                self._next_phase("RETRACT_FROM_TABLE")

        elif self.phase == "RETRACT_FROM_TABLE":
            action[20] = self.BJ5_TABLE
            action[15] = 1.0
            target_world = carton_pos.copy()
            target_world[2] += 0.28
            to_robot = self.ROBOT_BASE[:2] - target_world[:2]
            to_robot_norm = to_robot / (np.linalg.norm(to_robot) + 1e-8)
            target_world[:2] += to_robot_norm * 0.18
            target_local = self.world_to_arm_base(target_world, body_joints)
            new_joints, dist = self._move_right_toward(
                target_local, r_eef_pos, r_eef_quat, current_arm_14,
                step_size=0.012, n_iter=15
            )
            action[7:14] = new_joints
            self._log_debug(r_eef_pos, body_joints, target_world, "RETRACT")
            if dist < 0.05 or self.phase_step > 120:
                self._next_phase("ROTATE_TO_SCANNER")

        elif self.phase == "ROTATE_TO_SCANNER":
            action[20] = self._smooth_bj5(bj5, self.BJ5_SCANNER, 0.035)
            action[15] = 1.0
            action[7:14] = self.last_right_arm
            if abs(bj5 - self.BJ5_SCANNER) < 0.05 and self.phase_step > 20:
                self._next_phase("APPROACH_SCANNER")
            elif self.phase_step > 150:
                self._next_phase("APPROACH_SCANNER")

        elif self.phase == "APPROACH_SCANNER":
            action[20] = self.BJ5_SCANNER
            action[15] = 1.0
            target_world = self.SCANNER_POS.copy()
            target_world[2] += 0.15
            target_local = self.world_to_arm_base(target_world, body_joints)
            new_joints, dist = self._move_right_toward(
                target_local, r_eef_pos, r_eef_quat, current_arm_14,
                step_size=0.012, n_iter=15
            )
            action[7:14] = new_joints
            self._log_debug(r_eef_pos, body_joints, target_world, "APPROACH_SCANNER")
            if dist < 0.04 or self.phase_step > 400:
                self._next_phase("LOWER_TO_SCANNER")

        elif self.phase == "LOWER_TO_SCANNER":
            action[20] = self.BJ5_SCANNER
            action[15] = 1.0
            target_world = self.SCANNER_POS.copy()
            target_world[2] += 0.03
            target_local = self.world_to_arm_base(target_world, body_joints)
            new_joints, dist = self._move_right_toward(
                target_local, r_eef_pos, r_eef_quat, current_arm_14,
                step_size=0.006, n_iter=15
            )
            action[7:14] = new_joints
            self._log_debug(r_eef_pos, body_joints, target_world, "LOWER_SCANNER")
            if dist < 0.03 or self.phase_step > 200:
                self._next_phase("RELEASE_SCANNER")

        elif self.phase == "RELEASE_SCANNER":
            action[20] = self.BJ5_SCANNER
            action[7:14] = self.last_right_arm
            self.right_grip = 0.0
            action[15] = 0.0
            if self.phase_step >= 30:
                self._next_phase("LIFT_FROM_SCANNER_PRE")

        elif self.phase == "LIFT_FROM_SCANNER_PRE":
            action[20] = self.BJ5_SCANNER
            action[15] = 0.0
            target_world = self.SCANNER_POS.copy()
            target_world[2] += 0.13
            target_local = self.world_to_arm_base(target_world, body_joints)
            new_joints, dist = self._move_right_toward(
                target_local, r_eef_pos, r_eef_quat, current_arm_14,
                step_size=0.010, n_iter=15
            )
            action[7:14] = new_joints
            if dist < 0.04 or self.phase_step > 100:
                self._next_phase("WAIT_SETTLE")

        elif self.phase == "WAIT_SETTLE":
            action[20] = self.BJ5_SCANNER
            action[15] = 0.0
            action[7:14] = self.last_right_arm
            if self.phase_step >= 90:
                self._next_phase("LOWER_REGRASP")

        elif self.phase == "LOWER_REGRASP":
            action[20] = self.BJ5_SCANNER
            action[15] = 0.0
            target_world = self.SCANNER_POS.copy()
            target_world[2] += 0.03
            target_local = self.world_to_arm_base(target_world, body_joints)
            new_joints, dist = self._move_right_toward(
                target_local, r_eef_pos, r_eef_quat, current_arm_14,
                step_size=0.008, n_iter=15
            )
            action[7:14] = new_joints
            self._log_debug(r_eef_pos, body_joints, target_world, "LOWER_REGRASP")
            if dist < 0.03 or self.phase_step > 150:
                self._next_phase("REGRASP_SCANNER")

        elif self.phase == "REGRASP_SCANNER":
            action[20] = self.BJ5_SCANNER
            action[7:14] = self.last_right_arm
            self.right_grip = 1.0
            action[15] = 1.0
            if self.phase_step >= 35:
                self._next_phase("LIFT_FROM_SCANNER")

        elif self.phase == "LIFT_FROM_SCANNER":
            action[20] = self.BJ5_SCANNER
            action[15] = 1.0
            target_world = self.SCANNER_POS.copy()
            target_world[2] += 0.22
            target_local = self.world_to_arm_base(target_world, body_joints)
            new_joints, dist = self._move_right_toward(
                target_local, r_eef_pos, r_eef_quat, current_arm_14,
                step_size=0.012, n_iter=15
            )
            action[7:14] = new_joints
            self._log_debug(r_eef_pos, body_joints, target_world, "LIFT_FROM_SCANNER")
            if dist < 0.05 or self.phase_step > 150:
                self._next_phase("ROTATE_TO_BIN")

        elif self.phase == "ROTATE_TO_BIN":
            action[20] = self._smooth_bj5(bj5, self.BJ5_BIN, 0.035)
            action[15] = 1.0
            action[7:14] = self.last_right_arm
            if abs(bj5 - self.BJ5_BIN) < 0.05 and self.phase_step > 20:
                self._next_phase("APPROACH_BIN")
            elif self.phase_step > 100:
                self._next_phase("APPROACH_BIN")

        elif self.phase == "APPROACH_BIN":
            action[20] = self.BJ5_BIN
            action[15] = 1.0
            arm_world = self.arm_base_to_world(np.zeros(3), body_joints)
            to_bin = self.BIN_POS - arm_world
            to_bin_dir = to_bin / (np.linalg.norm(to_bin) + 1e-8)
            max_reach_world = arm_world + to_bin_dir * 0.65
            max_reach_world[2] = max(self.BIN_POS[2] + 0.20, max_reach_world[2])
            target_local = self.world_to_arm_base(max_reach_world, body_joints)
            new_joints, dist = self._move_right_toward(
                target_local, r_eef_pos, r_eef_quat, current_arm_14,
                step_size=0.012, n_iter=15
            )
            action[7:14] = new_joints
            self._log_debug(r_eef_pos, body_joints, max_reach_world, "APPROACH_BIN")
            if dist < 0.06 or self.phase_step > 300:
                self._next_phase("RELEASE_BIN")

        elif self.phase == "RELEASE_BIN":
            action[20] = self.BJ5_BIN
            action[7:14] = self.last_right_arm
            self.right_grip = 0.0
            action[15] = 0.0
            if self.phase_step >= 40:
                self._next_phase("RETURN_WAIST")

        elif self.phase == "RETURN_WAIST":
            action[20] = self._smooth_bj5(bj5, self.INIT_BODY[4], 0.035)
            action[15] = 0.0
            action[7:14] = self.last_right_arm
            if abs(bj5 - self.INIT_BODY[4]) < 0.1 and self.phase_step > 20:
                self._next_phase("DONE")
            elif self.phase_step > 100:
                self._next_phase("DONE")

        elif self.phase == "DONE":
            action[20] = self.INIT_BODY[4]
            action[15] = 0.0
            action[7:14] = self.last_right_arm

        return action


# ── Standalone test (outside Docker) ──────────────────────────────────
if __name__ == "__main__":
    policy = ScriptedSortingPolicy()

    print("=== Body FK Test ===")
    T = policy.body_fk(policy.INIT_BODY)
    arm_base_world = T[:3, 3] + policy.ROBOT_BASE
    print(f"arm_base_link world position (init): {arm_base_world}")
    print(f"arm_base_link rotation:\n{T[:3, :3]}")

    print("\n=== Reachability at different bj5 values ===")
    for bj5_val in [-2.0, -1.045, 0.0, 2.0]:
        body = policy.INIT_BODY.copy()
        body[4] = bj5_val
        T = policy.body_fk(body)
        arm_pos = T[:3, 3] + policy.ROBOT_BASE

        for name, world_pos in [("carton_v0", policy.CARTON_POSITIONS[0]),
                                  ("carton_v1", policy.CARTON_POSITIONS[1]),
                                  ("scanner", policy.SCANNER_POS),
                                  ("bin", policy.BIN_POS)]:
            local = policy.world_to_arm_base(world_pos, body)
            print(f"  bj5={bj5_val:+.2f} | {name:10s} local={local} "
                  f"dist={np.linalg.norm(local):.3f}")
        print()
