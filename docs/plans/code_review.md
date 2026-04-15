# Code Review & Modification Suggestions (Round 6)

**Date**: 2026-04-14
**Reviewer**: Local Claude Code (Evaluation)
**Target**: Cloud Claude Code reads this doc and modifies code

---

## Round 5 Diagnostic Results

Commit `1a28034`: Added real gripper world pose diagnostic in `_patched_step`.

**Diagnostic output reveals CRITICAL body FK error:**
```
[Init] Calibration at step 1:
  body_joints: [-1.04545222194, 1.34390352404, -0.31939525311, 0.0, 1.57]
  EEF local:   [ 0.3349, -0.0063, -0.4105]   (arm_base frame, from IKFKSolver)
  EEF world:   [ 0.934,   0.677,   0.896]     (from body_fk + ROBOT_BASE)

[Diag] step=1  real_grip=[0.4176, 0.6188, 1.2383]
[Diag] step=31 real_grip=[0.4175, 0.6171, 1.2375]  carton=[0.017,0.613,1.190] dist=0.4034
[Diag] step=61 real_grip=[0.3996, 0.6101, 1.2289]  carton dist=0.3854
...later steps: real dist INCREASES to 1.0+ as arm moves away
```

**FK vs Reality comparison at step 1:**
| | FK-computed (eef_w) | Real (gripper query) | Difference |
|---|---|---|---|
| x | 0.934 | 0.418 | **-0.516** |
| y | 0.677 | 0.619 | -0.058 |
| z | 0.896 | 1.238 | **+0.342** |

**Conclusion**: The `body_fk()` + `ROBOT_BASE` transform is fundamentally wrong. The FK-computed world position is ~0.6m away from the real position. When the policy uses `world_to_arm_base()` to convert the carton target to arm_base_link frame, it gives IK a completely wrong target. The arm moves in the wrong direction in reality.

---

## BUG 11 Fix: Closed-Loop Correction Using Real Gripper Position

**Root cause**: `BODY_JOINTS_URDF` parameters and/or `ROBOT_BASE` values are incorrect. Rather than guessing correct values, we bypass the broken body FK entirely by using runtime correction from the real gripper position.

**Algorithm (correction offset):**

Each step:
1. Query real gripper world position from `/genie/gripper_r_center_link`
2. Compute FK gripper world position: `fk_world = arm_base_to_world(eef_local, body_joints)`
3. Compute correction: `correction = real_world - fk_world`
4. For any world target, adjust: `adjusted_target = target_world - correction`
5. Convert to arm_base: `target_local = world_to_arm_base(adjusted_target, body_joints)`
6. Existing IK solver works unchanged

**Why this works**: The correction cancels the FK translation error. If FK rotation is also slightly wrong, the error is proportional to `(R_correct - R_wrong) @ (target_local - current_local)`, which is small for nearby targets. Since we recompute correction every step, the arm converges to the correct position in a closed loop.

---

## Implementation Plan

### File 1: `scripts/run_sorting_benchmark.py`

#### Change 1A: Create shared state dict and pass to policy

In `_create_env_pi`, after creating the policy and setting scene positions, create a shared dict and attach it to the policy:

```python
# After the line: self.policy.set_scene_positions(...)
# Add shared state for real-time gripper feedback
_shared_state = {"real_gripper_world": None}
if hasattr(self, 'policy') and self.policy is not None:
    self.policy._shared_state = _shared_state
    print("[Patch] Shared state attached to policy")
```

#### Change 1B: Query real gripper position EVERY step (not just every 30)

In `_patched_step`, move the gripper query outside the `% 30` block so it runs every step. Keep the print logging at every 30 steps.

```python
def _patched_step(action):
    result = _orig_step(action)
    _step_counter[0] += 1

    # Hold bj1-bj4 (existing code unchanged)
    if not _body_indices_cache:
        if hasattr(_env, 'robot_joint_indices'):
            _body_indices_cache.extend(
                _env.robot_joint_indices[v] for v in _body_names
            )
            print(f"[Patch] bj1-bj4 hold active: indices={_body_indices_cache}")
    if _body_indices_cache:
        _env.api_core.set_joint_positions(
            [float(v) for v in _body_hold],
            joint_indices=_body_indices_cache,
            is_trajectory=True,
        )

    # Query real gripper world position EVERY step for closed-loop control
    try:
        rp, _ = _env.api_core.get_obj_world_pose(
            "/genie/gripper_r_center_link"
        )
        _shared_state["real_gripper_world"] = [
            float(rp[0]), float(rp[1]), float(rp[2])
        ]
    except Exception:
        pass  # keep last known position

    # Diagnostic logging every 30 steps
    if _step_counter[0] % 30 == 1:
        rg = _shared_state.get("real_gripper_world")
        if rg:
            msg = (f"[Diag] step={_step_counter[0]} "
                   f"real_grip=[{rg[0]:.4f},{rg[1]:.4f},{rg[2]:.4f}]")
            if _diag_carton_name:
                for par in ["/Workspace/Objects"]:
                    try:
                        cp, _ = _env.api_core.get_obj_world_pose(
                            f"{par}/{_diag_carton_name}"
                        )
                        import math
                        d = math.sqrt(sum(
                            (rg[i] - float(cp[i])) ** 2 for i in range(3)
                        ))
                        msg += (f" carton=[{cp[0]:.3f},{cp[1]:.3f},"
                                f"{cp[2]:.3f}] dist={d:.4f}")
                        break
                    except Exception:
                        continue
            print(msg)

    return result
```

Note: `_shared_state` must be captured in the closure. It's defined in `_create_env_pi` before `_patched_step`.

#### Change 1C: Ensure _shared_state is in scope for _patched_step

The `_shared_state` dict should be defined BEFORE the `_patched_step` function, in the same scope as the other closure variables (`_body_indices_cache`, `_step_counter`, etc.). Place it right after `_diag_carton_name = target_carton_name`:

```python
_diag_carton_name = target_carton_name
_shared_state = {"real_gripper_world": None}

# Pass shared state to policy
if hasattr(self, 'policy') and self.policy is not None:
    self.policy._shared_state = _shared_state
    print("[Patch] Shared state attached to policy")
```

### File 2: `scripts/scripted_sorting_policy.py`

#### Change 2A: Add correction computation method

Add a new method to `ScriptedSortingPolicy`:

```python
def _get_fk_correction(self, obs: dict) -> np.ndarray:
    """Compute world-frame correction vector: real_gripper - fk_gripper.
    
    This correction compensates for errors in body_fk / ROBOT_BASE.
    Applied to world targets before converting to arm_base frame.
    """
    if not hasattr(self, '_shared_state') or self._shared_state is None:
        return np.zeros(3)
    
    real_grip = self._shared_state.get("real_gripper_world")
    if real_grip is None:
        return np.zeros(3)
    
    real_grip = np.array(real_grip)
    fk_grip = self.arm_base_to_world(obs["r_eef_pos"], obs["body"])
    correction = real_grip - fk_grip
    return correction
```

#### Change 2B: Add corrected world_to_arm_base method

```python
def _corrected_world_to_arm(
    self, world_pos: np.ndarray, obs: dict
) -> np.ndarray:
    """Convert world position to arm_base frame with FK correction."""
    correction = self._get_fk_correction(obs)
    adjusted = world_pos - correction
    return self.world_to_arm_base(adjusted, obs["body"])
```

#### Change 2C: Replace all world_to_arm_base calls in phase handlers

Replace every instance of:
```python
target_l = self.world_to_arm_base(target_w, obs["body"])
```
with:
```python
target_l = self._corrected_world_to_arm(target_w, obs)
```

This occurs in these methods:
1. `_phase_approach` (line ~517)
2. `_phase_grasp` (lines ~541, ~565)
3. `_phase_move_to_scanner` (lines ~597, ~613)
4. `_phase_regrasp` (lines ~648, ~665, ~689)
5. `_phase_move_to_bin` (line ~722)
6. `_compute_bj5_for_target` (line ~208)

**Note on `_compute_bj5_for_target`**: This is called at init when no real gripper data exists yet. It should keep using the uncorrected `world_to_arm_base` since it only needs approximate angles. Do NOT change this method.

#### Change 2D: Update _log to show correction

Modify `_log()` to also print the correction vector:

```python
def _log(self, obs: dict, target_world=None, label=""):
    if self.step_count % 30 != 0:
        return
    correction = self._get_fk_correction(obs)
    # Use real gripper position for logging (more accurate)
    real_grip = (self._shared_state.get("real_gripper_world")
                 if hasattr(self, '_shared_state') and self._shared_state
                 else None)
    if real_grip is not None:
        rg = np.array(real_grip)
        msg = (f"[s={self.step_count}] {self.phase}:{self.sub_step} "
               f"eef_real=[{rg[0]:.3f},{rg[1]:.3f},{rg[2]:.3f}]"
               f" corr=[{correction[0]:.3f},{correction[1]:.3f},{correction[2]:.3f}]")
    else:
        eef_w = self.arm_base_to_world(obs["r_eef_pos"], obs["body"])
        msg = (f"[s={self.step_count}] {self.phase}:{self.sub_step} "
               f"eef_w=[{eef_w[0]:.3f},{eef_w[1]:.3f},{eef_w[2]:.3f}]")
    if target_world is not None:
        target_l = self._corrected_world_to_arm(target_world, obs)
        d = np.linalg.norm(target_l - obs["r_eef_pos"])
        msg += f" tgt=[{target_world[0]:.3f},{target_world[1]:.3f},{target_world[2]:.3f}] d={d:.3f}"
    if label:
        msg += f" [{label}]"
    print(msg)
```

#### Change 2E: Also update _log_init to show correction on first step

In `_log_init`, after the existing calibration output, add:

```python
# Show FK correction if available
correction = self._get_fk_correction(obs)
if np.linalg.norm(correction) > 0.01:
    print(f"  FK correction: [{correction[0]:.4f}, {correction[1]:.4f}, {correction[2]:.4f}]")
    print(f"  ||correction||: {np.linalg.norm(correction):.4f}m")
```

#### Change 2F: Initialize _shared_state in __init__ and reset

In `__init__`:
```python
self._shared_state = None  # set by benchmark runner
```

In `reset()`:
```python
# Do NOT reset _shared_state here — it's a reference to the runner's dict
```

---

## Expected Behavior After Fix

1. Step 1: correction computed as `[0.418, 0.619, 1.238] - [0.934, 0.677, 0.896] = [-0.516, -0.058, +0.342]`
2. When computing carton target local: `adjusted_carton = [0.017, 0.613, 1.19] - [-0.516, -0.058, +0.342] = [0.533, 0.671, 0.848]`
3. `target_local = world_to_arm_base([0.533, 0.671, 0.848], body)` — this should give IK a target that moves the real gripper toward the carton
4. Each step, correction updates to track changes in body configuration
5. APPROACH should converge with real distance decreasing (not increasing)
6. Follow evaluator should trigger when real gripper enters carton AABB (±0.1m)

---

## Verification Checklist

- [ ] `[Diag]` shows correction vector at init: expect ~[-0.5, -0.06, +0.34]
- [ ] `[Diag]` real_grip-carton dist DECREASES over APPROACH phase (was increasing before)
- [ ] `Follow` evaluator score > 0 (gripper enters carton AABB)
- [ ] GRASP still works (gripper closes around carton)
- [ ] If correction drifts significantly during operation, may need per-step update frequency tuning

---

## Priority

1. **Implement closed-loop correction** (BUG 11 fix) - Changes 1A-1C and 2A-2F
2. **Do not change anything else** - all other code is working correctly
