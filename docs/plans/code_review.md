# Code Review & Modification Suggestions (Round 7)

**Date**: 2026-04-16
**Reviewer**: Local Claude Code (Evaluation)
**Target**: Cloud Claude Code reads this doc and modifies code

---

## Round 6 Result: All 0 — Correction Approach Completely Failed

Commit `61af3b8`: Closed-loop FK correction via `_corrected_world_to_arm`.

**The correction approach makes things WORSE, not better.** The arm moves AWAY from the carton.

### Evidence

```
Step  1: real_grip=[0.418, 0.619, 1.238]  ← INSIDE Follow AABB!
Step 60: real_grip=[0.201, 0.682, 1.324]  ← arm starts moving AWAY
Step 120: real_grip=[-0.079, 0.582, 1.410] ← now x is NEGATIVE
Step 300+: real_grip=[0.098, 0.453, 1.455] ← stabilized 0.35m away
```

**Critical discovery**: At step 1 (init pose), the real gripper is ALREADY INSIDE the Follow AABB for the first carton!

```
Carton: [0.328, 0.687, 1.183], bbox=[0.2,0.2,0.2]
AABB:   x[0.228, 0.428], y[0.587, 0.787], z[1.083, 1.283]
Gripper: [0.418, 0.619, 1.238]
  x=0.418 ∈ [0.228, 0.428] ✓
  y=0.619 ∈ [0.587, 0.787] ✓  
  z=1.238 ∈ [1.083, 1.283] ✓
```

But then APPROACH starts moving the arm + rotating bj5 toward 3.048 (also computed with broken FK), and the gripper leaves the AABB.

### Root Cause Analysis

The `_corrected_world_to_arm` correction offset approach fails because the body FK **rotation** is wrong, not just the translation. A translation-only correction cannot fix a rotation error. When the arm moves, the correction vector changes (from [-0.567,-0.138,0.327] to [-0.577,-0.320,0.276]), causing unstable feedback that makes the arm diverge.

Additionally, `_compute_bj5_for_target` also uses the broken body FK, producing wrong bj5 angles (3.048 instead of ~1.57).

---

## BUG 12 [CRITICAL]: bj5 computation uses broken body FK, produces wrong waist angle

`_compute_bj5_for_target()` sweeps bj5 values and calls `world_to_arm_base()` to find the best angle. Since body_fk rotation is wrong, the computed bj5 is completely wrong (3.048 vs ~1.57), causing the waist to rotate away from the carton.

---

## Fix Strategy: Two-Phase Approach

### Phase A: Bypass broken FK for APPROACH (immediate fix)

Since the gripper is already inside the Follow AABB at init, the simplest fix is: **don't move the arm or rotate bj5 during APPROACH**. Just hold the init pose.

### Phase B: Find the correct world→arm_base transform (for GRASP and beyond)

We need the actual arm_base_link world pose from the simulation. This requires:
1. Finding the correct prim name by enumerating `/genie/` children
2. Querying the prim's world pose (position + quaternion)
3. Using it as the correct T_world_to_arm

---

## Implementation Plan

### File 1: `scripts/run_sorting_benchmark.py`

#### Change 1A: Add prim enumeration diagnostic at step 1

In `_patched_step`, at `_step_counter[0] == 1`, enumerate all children of `/genie/` and log link names. This tells us the arm_base_link prim name for the next round.

Replace the existing diagnostic section with:

```python
def _patched_step(action):
    result = _orig_step(action)
    _step_counter[0] += 1

    # Hold bj1-bj4 (unchanged)
    if not _body_indices_cache:
        if hasattr(_env, 'robot_joint_indices'):
            _body_indices_cache.extend(
                _env.robot_joint_indices[v] for v in _body_names
            )
            print(f"[Patch] bj1-bj4 hold active: "
                  f"indices={_body_indices_cache}")
    if _body_indices_cache:
        _env.api_core.set_joint_positions(
            [float(v) for v in _body_hold],
            joint_indices=_body_indices_cache,
            is_trajectory=True,
        )

    # Query real gripper EVERY step for closed-loop
    try:
        rp, _ = _env.api_core.get_obj_world_pose(
            "/genie/gripper_r_center_link"
        )
        _shared_state["real_gripper_world"] = [
            float(rp[0]), float(rp[1]), float(rp[2])
        ]
    except Exception:
        pass

    # Step 1: enumerate prims and try arm_base candidates
    if _step_counter[0] == 1:
        # Enumerate children of /genie/ to find link names
        try:
            stage = _env.api_core._stage
            genie_prim = stage.GetPrimAtPath("/genie")
            if genie_prim.IsValid():
                arm_links = []
                for child in genie_prim.GetChildren():
                    name = child.GetName()
                    if ("arm" in name.lower()
                            or "base" in name.lower()
                            or "link" in name.lower()):
                        arm_links.append(name)
                print(f"[Enum] /genie/ arm/base/link children "
                      f"({len(arm_links)}): {arm_links}")
            else:
                print("[Enum] /genie/ prim not valid")
        except Exception as e:
            print(f"[Enum] Prim enumeration failed: {e}")

        # Try arm_base_link candidates
        _arm_base_candidates = [
            "/genie/arm_r_base_link",
            "/genie/right_arm_base_link",
            "/genie/arm_base_link",
            "/genie/torso_arm_r_base_link",
            "/genie/link_arm_r_base",
            "/genie/arm_r_link0",
        ]
        for cand in _arm_base_candidates:
            try:
                ab_pos, ab_rot = _env.api_core.get_obj_world_pose(cand)
                print(f"[ArmBase] FOUND: {cand}")
                print(f"  pos=[{ab_pos[0]:.4f},{ab_pos[1]:.4f},"
                      f"{ab_pos[2]:.4f}]")
                print(f"  rot=[{ab_rot[0]:.4f},{ab_rot[1]:.4f},"
                      f"{ab_rot[2]:.4f},{ab_rot[3]:.4f}]")
                _shared_state["arm_base_prim"] = cand
                _shared_state["arm_base_pos"] = [
                    float(ab_pos[i]) for i in range(3)
                ]
                _shared_state["arm_base_rot"] = [
                    float(ab_rot[i]) for i in range(4)
                ]
                break
            except Exception:
                continue
        if "arm_base_prim" not in _shared_state:
            print("[ArmBase] No candidate found! "
                  "Check [Enum] output for correct name.")

        # Also query left gripper for calibration
        try:
            lp, _ = _env.api_core.get_obj_world_pose(
                "/genie/gripper_l_center_link"
            )
            print(f"[Calib] left_grip_world="
                  f"[{lp[0]:.4f},{lp[1]:.4f},{lp[2]:.4f}]")
            _shared_state["left_gripper_world"] = [
                float(lp[i]) for i in range(3)
            ]
        except Exception as e:
            print(f"[Calib] left gripper query failed: {e}")

    # Diagnostic logging every 30 steps
    if _step_counter[0] % 30 == 1:
        rg = _shared_state.get("real_gripper_world")
        if rg:
            msg = (f"[Diag] step={_step_counter[0]}"
                   f" real_grip=[{rg[0]:.4f},"
                   f"{rg[1]:.4f},{rg[2]:.4f}]")
            if _diag_carton_name:
                for par in ["/Workspace/Objects"]:
                    try:
                        cp, _ = (
                            _env.api_core.get_obj_world_pose(
                                f"{par}/{_diag_carton_name}"
                            )
                        )
                        import math
                        d = math.sqrt(sum(
                            (rg[i] - float(cp[i])) ** 2
                            for i in range(3)
                        ))
                        msg += (
                            f" carton=[{cp[0]:.3f},"
                            f"{cp[1]:.3f},{cp[2]:.3f}]"
                            f" dist={d:.4f}"
                        )
                        break
                    except Exception:
                        continue
            print(msg)

    # If arm_base_prim was found, update its world pose each step
    if _shared_state.get("arm_base_prim"):
        try:
            ab_pos, ab_rot = _env.api_core.get_obj_world_pose(
                _shared_state["arm_base_prim"]
            )
            _shared_state["arm_base_pos"] = [
                float(ab_pos[i]) for i in range(3)
            ]
            _shared_state["arm_base_rot"] = [
                float(ab_rot[i]) for i in range(4)
            ]
        except Exception:
            pass

    return result
```

### File 2: `scripts/scripted_sorting_policy.py`

#### Change 2A: Replace `_corrected_world_to_arm` with arm_base prim-based transform

Replace the existing `_get_fk_correction` and `_corrected_world_to_arm` methods with:

```python
def _get_real_arm_base_T(self) -> np.ndarray | None:
    """Get real arm_base → world transform from sim query.
    
    Returns 4x4 homogeneous transform, or None if not available.
    """
    if self._shared_state is None:
        return None
    ab_pos = self._shared_state.get("arm_base_pos")
    ab_rot = self._shared_state.get("arm_base_rot")
    if ab_pos is None or ab_rot is None:
        return None
    
    pos = np.array(ab_pos)
    # Quaternion wxyz → rotation matrix
    qw, qx, qy, qz = ab_rot
    if R is not None:
        rot_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
    else:
        rot_mat = np.eye(3)
    
    # Apply ARM_BASE_RPY offset (arm_base frame is rotated
    # relative to the link frame)
    rpy_rot = self._rot('x', self.ARM_BASE_RPY[0])
    rot_mat = rot_mat @ rpy_rot
    
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos
    return T

def _corrected_world_to_arm(
    self, world_pos: np.ndarray, obs: dict,
) -> np.ndarray:
    """world → arm_base_link using real sim transform if available."""
    T = self._get_real_arm_base_T()
    if T is not None:
        T_inv = np.linalg.inv(T)
        return (T_inv @ np.append(world_pos, 1.0))[:3]
    # Fallback to body FK
    return self.world_to_arm_base(world_pos, obs["body"])

def _corrected_arm_to_world(
    self, local_pos: np.ndarray, obs: dict,
) -> np.ndarray:
    """arm_base_link → world using real sim transform if available."""
    T = self._get_real_arm_base_T()
    if T is not None:
        return (T @ np.append(local_pos, 1.0))[:3]
    return self.arm_base_to_world(local_pos, obs["body"])
```

**IMPORTANT**: The ARM_BASE_RPY offset may or may not be needed depending on how the prim's frame aligns with IKFKSolver's arm_base frame. If the results are still wrong after adding this, try removing the `rpy_rot` line:
```python
# rot_mat = rot_mat @ rpy_rot  # try without this if results are wrong
```

#### Change 2B: Fix bj5 computation — use real gripper position instead of FK

Replace `_compute_bj5_for_target()` with a version that doesn't depend on body FK:

```python
def _compute_bj5_for_target(self, target_world: np.ndarray) -> float:
    """Find bj5 angle that points the robot toward the target.
    
    Simple geometric approach: compute the angle from robot base
    to target in world xy plane. No body FK needed.
    """
    dx = target_world[0] - self.ROBOT_BASE[0]
    dy = target_world[1] - self.ROBOT_BASE[1]
    # atan2 gives angle in world frame
    # bj5=0 means facing forward (+x), bj5=pi/2 means facing left (+y)
    angle = np.arctan2(dy, dx)
    
    # bj5 convention: at init bj5=1.57 (pi/2), robot faces +y
    # So bj5 = angle (since bj5 IS the yaw angle)
    # Clamp to joint limits
    bj5 = np.clip(angle, -3.1, 3.1)
    return bj5
```

**Note**: The relationship between bj5 and world-frame yaw might have an offset. At init bj5=1.57 (≈π/2), the carton is roughly in the +y direction. atan2(0.687-0.093, 0.328-0.245) = atan2(0.594, 0.083) ≈ 1.43 ≈ π/2. So bj5 ≈ atan2(dy, dx) is approximately correct.

#### Change 2C: APPROACH — hold arm position, only adjust bj5 if needed

Modify `_phase_approach` to HOLD the arm at init position instead of moving it:

```python
def _phase_approach(self, obs: dict) -> np.ndarray:
    # Check if we have real gripper position and it's already in AABB
    real_grip = None
    if (self._shared_state is not None
            and self._shared_state.get("real_gripper_world")):
        real_grip = np.array(self._shared_state["real_gripper_world"])
    
    # Compute distance to carton in world frame
    if real_grip is not None:
        world_dist = np.linalg.norm(real_grip - self._carton_pos)
    else:
        world_dist = float('inf')
    
    # If gripper is already close to carton (within ~0.15m),
    # just hold position — don't risk moving away
    if world_dist < 0.2:
        # Hold init arm position, optionally fine-tune bj5
        bj5_target = self._bj5_table
        new_bj5 = self._smooth_bj5(obs["bj5"], bj5_target, self.BJ5_SPEED)
        action = self._build_action(
            obs["left_arm"], self.last_right_arm, new_bj5,
        )
        self._log(obs, self._carton_pos, "holding_near_carton")
        
        if self.sub_step > 100:
            self._set_phase("GRASP")
        return action
    
    # Otherwise, try to move toward carton
    bj5_target = self._bj5_table
    new_bj5 = self._smooth_bj5(obs["bj5"], bj5_target, self.BJ5_SPEED)
    
    target_w = self._carton_pos.copy()
    target_w[2] += self.APPROACH_HEIGHT
    target_l = self._corrected_world_to_arm(target_w, obs)
    new_joints, dist = self._move_right_toward(
        target_l, obs["r_eef_pos"], obs["r_eef_quat"],
        obs["arm_14"], step_size=self.EEF_STEP_FAST,
    )
    action = self._build_action(obs["left_arm"], new_joints, new_bj5)
    self._log(obs, target_w, "approach_moving")
    
    if dist < 0.04 or self.sub_step > 500:
        self._set_phase("GRASP")
    return action
```

#### Change 2D: Update `_log` to show world-frame distance

```python
def _log(self, obs, target_world=None, label=""):
    if self.step_count % 30 != 0:
        return
    real_grip = None
    if (self._shared_state is not None
            and self._shared_state.get("real_gripper_world")):
        real_grip = np.array(self._shared_state["real_gripper_world"])
    if real_grip is not None:
        msg = (f"[s={self.step_count}] {self.phase}:{self.sub_step} "
               f"eef_real=[{real_grip[0]:.3f},{real_grip[1]:.3f},"
               f"{real_grip[2]:.3f}]")
    else:
        eef_w = self.arm_base_to_world(obs["r_eef_pos"], obs["body"])
        msg = (f"[s={self.step_count}] {self.phase}:{self.sub_step} "
               f"eef_w=[{eef_w[0]:.3f},{eef_w[1]:.3f},{eef_w[2]:.3f}]")
    if target_world is not None:
        if real_grip is not None:
            d_world = np.linalg.norm(real_grip - target_world)
            msg += (f" tgt=[{target_world[0]:.3f},{target_world[1]:.3f},"
                    f"{target_world[2]:.3f}] d_world={d_world:.3f}")
        else:
            tgt_l = self.world_to_arm_base(target_world, obs["body"])
            d = np.linalg.norm(tgt_l - obs["r_eef_pos"])
            msg += (f" tgt=[{target_world[0]:.3f},{target_world[1]:.3f},"
                    f"{target_world[2]:.3f}] d_fk={d:.3f}")
    if label:
        msg += f" [{label}]"
    print(msg)
```

#### Change 2E: Remove old `_get_fk_correction()` method

Delete the old `_get_fk_correction` method entirely. It is replaced by `_get_real_arm_base_T` above.

#### Change 2F: Update all arm_base_to_world calls in logging

In `_log_init`, use `_corrected_arm_to_world` for the FK world output:

```python
eef_w = self._corrected_arm_to_world(obs["r_eef_pos"], obs)
```

---

## What to Expect

### If arm_base_link prim IS found:
- `[ArmBase] FOUND: /genie/arm_r_base_link` (or similar)
- `_corrected_world_to_arm` uses the real transform
- APPROACH should converge correctly (real distance decreases)
- Follow should pass

### If arm_base_link prim is NOT found:
- `[Enum]` output shows available link names → use for next round
- `_corrected_world_to_arm` falls back to body FK (still broken for arm movement)
- But APPROACH holds arm when gripper is close → Follow may pass for nearby cartons

### bj5 fix:
- `bj5_table` should be ~1.43 (instead of 3.048) for the first carton
- Waist rotates only slightly from init (1.57 → 1.43), keeping gripper near carton

---

## Verification Checklist

- [ ] `[Enum]` output shows available link names under `/genie/`
- [ ] `[ArmBase] FOUND` or prim names to try in next round
- [ ] `bj5_table` is ≈1.4 (not 3.048) for the first carton
- [ ] APPROACH: gripper stays near carton (real dist < 0.2m, ideally < 0.1m)
- [ ] Follow score > 0 for at least one episode
- [ ] `[Calib] left_grip_world` printed (for future rotation calibration)

---

## Priority

1. **Change 1A**: Prim enumeration + arm_base query (diagnostic + fix)
2. **Change 2B**: Fix bj5 computation (remove FK dependency)
3. **Change 2A**: Real arm_base transform (replace broken correction)
4. **Change 2C**: APPROACH hold-if-close logic
5. **Change 2D-2F**: Logging improvements
