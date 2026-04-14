#!/usr/bin/env python3
"""
Diagnostic script to verify IK/FK accuracy inside Docker.

Run with: /isaac-sim/python.sh /workspace/genie_sim_RL/scripts/diagnose_ik_fk.py
"""

import sys
import os
sys.path.insert(0, "/workspace/genie_sim/source")

import numpy as np
from scipy.spatial.transform import Rotation as R
from geniesim.utils.ikfk_utils import IKFKSolver, xyzrpy_to_xyzquat

# G2_STATES_4 initial values
INIT_BODY = [1.57, 0.0, -0.31939525311, 1.34390352404, -1.04545222194]
INIT_LEFT_ARM = [0.739033, -0.717023, -1.524419, -1.537612, 0.27811, -0.925845, -0.839257]
INIT_RIGHT_ARM = [-0.739033, -0.717023, 1.524419, -1.537612, -0.27811, -0.925845, 0.839257]

ROBOT_BASE = np.array([0.24469, 0.09325, 0.0])


def compute_right_ik_multi(solver, target_pos, target_rpy, current_arm_14, n_iter=1):
    """Compute right arm IK with multiple solver iterations."""
    eef = solver.compute_eef(current_arm_14)
    lp = eef["left"][:3]
    lq = eef["left"][3:7]
    lr = R.from_quat([lq[1], lq[2], lq[3], lq[0]]).as_euler('xyz')
    left_xyzrpy = np.concatenate([lp, lr])
    right_xyzrpy = np.concatenate([target_pos, target_rpy])
    eef_action = np.concatenate([left_xyzrpy, right_xyzrpy, [0.0, 0.0]])

    # Repeat same action n_iter times for solver convergence
    eef_actions = [eef_action] * n_iter
    result = solver.eef_actions_to_joint(eef_actions, current_arm_14, [0, 0])
    return np.array(result[-1][:14])  # take last (most converged) result


def main():
    arm_init = INIT_LEFT_ARM + INIT_RIGHT_ARM
    solver = IKFKSolver(arm_init, [0, 0, 0], INIT_BODY, robot_cfg="G2_omnipicker")
    print("[OK] IK/FK solver initialized\n")

    # === 1. FK for initial arm config ===
    print("=" * 60)
    print("1. FK for initial right arm configuration")
    print("=" * 60)
    eef = solver.compute_eef(arm_init)
    r_pos = np.array(eef["right"][:3])
    r_quat = np.array(eef["right"][3:7])
    print(f"  Right EEF pos (arm_base): {r_pos}")
    print(f"  Right EEF quat (qw,qx,qy,qz): {r_quat}")
    r_rot = R.from_quat([r_quat[1], r_quat[2], r_quat[3], r_quat[0]])
    r_rpy = r_rot.as_euler('xyz')
    r_mat = r_rot.as_matrix()
    print(f"  Right EEF rpy: {r_rpy}")
    print(f"  Right EEF Z-axis: {r_mat[:, 2]}")
    print(f"  EEF reach: {np.linalg.norm(r_pos):.4f}m")

    gripper_center = r_pos + r_mat @ np.array([0, 0, 0.14308])
    print(f"  Gripper center: {gripper_center}")
    print(f"  Gripper center reach: {np.linalg.norm(gripper_center):.4f}m")

    # === 2. Multi-iteration IK test ===
    print("\n" + "=" * 60)
    print("2. Multi-iteration IK accuracy test")
    print("=" * 60)

    current_arm_14 = np.array(arm_init, dtype=np.float32)
    current_eef = solver.compute_eef(current_arm_14)
    current_r_quat = np.array(current_eef["right"][3:7])
    current_r_rpy = R.from_quat([current_r_quat[1], current_r_quat[2],
                                   current_r_quat[3], current_r_quat[0]]).as_euler('xyz')

    targets = [
        ("close",       [0.35, -0.15, -0.30]),
        ("medium",      [0.40, -0.20, -0.20]),
        ("far",         [0.55, -0.30, -0.10]),
        ("very_far",    [0.70, -0.20,  0.00]),
        ("reach_limit", [0.80, -0.10,  0.10]),
    ]

    for n_iter in [1, 5, 10, 20, 50]:
        print(f"\n  --- n_iter = {n_iter} ---")
        for name, target_pos in targets:
            target_pos_np = np.array(target_pos, dtype=np.float32)
            try:
                joints = compute_right_ik_multi(
                    solver, target_pos_np, current_r_rpy,
                    current_arm_14, n_iter=n_iter
                )
                achieved_eef = solver.compute_eef(joints)
                achieved_pos = np.array(achieved_eef["right"][:3])
                error = np.linalg.norm(achieved_pos - target_pos_np)
                status = "OK" if error < 0.01 else "WARN" if error < 0.05 else "FAIL"
                print(f"    {name:13s}: err={error:.4f}m {status}")
            except Exception as e:
                print(f"    {name:13s}: EXCEPTION - {e}")

    # === 3. Incremental approach with multi-iter ===
    print("\n" + "=" * 60)
    print("3. Incremental approach with multi-iteration IK (n_iter=10)")
    print("=" * 60)

    final_target = np.array([0.55, -0.30, -0.10], dtype=np.float32)
    cur_arm = np.array(arm_init, dtype=np.float32)
    step_size = 0.02
    n_iter = 10

    for step in range(50):
        cur_eef = solver.compute_eef(cur_arm)
        cur_pos = np.array(cur_eef["right"][:3])
        cur_quat = np.array(cur_eef["right"][3:7])
        cur_rpy = R.from_quat([cur_quat[1], cur_quat[2], cur_quat[3],
                                cur_quat[0]]).as_euler('xyz')

        error_vec = final_target - cur_pos
        dist = np.linalg.norm(error_vec)
        if dist < 0.005:
            print(f"  Step {step}: CONVERGED dist={dist:.5f}")
            break

        if dist > step_size:
            move = error_vec / dist * step_size
        else:
            move = error_vec
        next_pos = (cur_pos + move).astype(np.float32)

        try:
            new_joints = compute_right_ik_multi(
                solver, next_pos, cur_rpy, cur_arm, n_iter=n_iter
            )
            cur_arm = new_joints
            if step % 5 == 0:
                print(f"  Step {step:2d}: dist={dist:.4f} pos=[{cur_pos[0]:.3f}, "
                      f"{cur_pos[1]:.3f}, {cur_pos[2]:.3f}]")
        except Exception as e:
            print(f"  Step {step}: IK error - {e}")
            break
    else:
        cur_eef = solver.compute_eef(cur_arm)
        final_pos = np.array(cur_eef["right"][:3])
        final_dist = np.linalg.norm(final_target - final_pos)
        print(f"  After 50 steps: dist={final_dist:.5f} pos={final_pos}")

    # === 4. Test with syncing after each solve ===
    print("\n" + "=" * 60)
    print("4. Direct solver access: multiple solve() calls")
    print("=" * 60)

    # Access the right solver directly
    right_solver = solver.right_solver
    right_solver.sync_target_with_joints(np.array(INIT_RIGHT_ARM, dtype=np.float32))

    target_pos = np.array([0.55, -0.30, -0.10], dtype=np.float32)
    target_quat = xyzrpy_to_xyzquat(
        np.concatenate([target_pos, current_r_rpy])
    )

    right_solver.update_target_quat(
        target_pos=target_quat[:3].astype(np.float32),
        target_quat=target_quat[3:].astype(np.float32),
    )

    for i in range(100):
        joints = right_solver.solve()
        if i % 20 == 0 or i == 99:
            fk_mat = np.asarray(
                right_solver.compute_fk(joints), dtype=np.float64
            ).reshape(4, 4)
            pos = fk_mat[:3, 3]
            error = np.linalg.norm(pos - target_pos)
            print(f"  Iter {i:3d}: pos=[{pos[0]:.4f}, {pos[1]:.4f}, "
                  f"{pos[2]:.4f}] err={error:.5f}m")

    # === 5. Max reach with solver ===
    print("\n" + "=" * 60)
    print("5. Maximum arm reach (extended configs)")
    print("=" * 60)

    max_reach = 0
    best_config = None
    test_configs = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.5, 0.0, -0.3, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5, -1.0, 0.5, -0.3, 0.0, 0.0, 0.0],
        [0.5, -1.5, 1.0, 0.0, -0.5, 0.0, 0.0],
        [-1.0, -1.5, 1.5, 0.5, 0.0, 0.5, 0.0],
        [0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0],
        # Try to maximize X reach (arm straight out in X direction)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, -0.3, 0.5, -0.2, 0.0, 0.0, 0.0],
        [-0.2, -0.3, -0.5, -0.2, 0.0, 0.0, 0.0],
    ]

    for config in test_configs:
        test_arm = INIT_LEFT_ARM + config
        try:
            test_eef = solver.compute_eef(test_arm)
            pos = np.array(test_eef["right"][:3])
            reach = np.linalg.norm(pos)
            quat = np.array(test_eef["right"][3:7])
            rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
            gc = pos + rot @ np.array([0, 0, 0.14308])
            gc_reach = np.linalg.norm(gc)
            if reach > max_reach:
                max_reach = reach
                best_config = config
            print(f"  Config {config} -> EEF reach={reach:.4f} "
                  f"gripper={gc_reach:.4f}")
        except Exception:
            pass

    print(f"\n  Max EEF reach: {max_reach:.4f}m")
    print(f"  Best config: {best_config}")

    print("\n[Done] Diagnostic complete.")


if __name__ == "__main__":
    main()
