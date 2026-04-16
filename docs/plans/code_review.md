# Code Review & Modification Suggestions (Round 9)

**Date**: 2026-04-16
**Reviewer**: Local Claude Code (Evaluation)
**Target**: Cloud Claude Code reads this doc and modifies code

---

## Round 8 Result: FOLLOW = 1.0 for yellow carton! Average = 0.0833

### Scores (8 episodes)

| Episode | Carton | Follow | PickUp | Inside | Upright | PickUp2 | Inside2 |
|---------|--------|--------|--------|--------|---------|---------|---------|
| 1–4     | Yellow | **1.0** | 0 | 0 | 0 | 0 | 0 |
| 5–8     | Black  | 0      | 0 | 0 | 0 | 0 | 0 |

**Average: 0.0833** (4 Follow successes × 1 point / 48 total)

### What Works Now

1. **Real-time carton position tracking**: `[Policy] Carton pos updated: [0.328,0.687,1.074] → [0.302,0.700,0.779] (diff=0.296m)` — BUG 13 FIXED!
2. **Follow = 1.0 for all 4 yellow carton episodes!** Gripper enters AABB at step 180 (z=0.860)
3. **Arm converges initially**: d_world 0.480 → 0.234 → 0.115 → 0.105 in APPROACH
4. **bj5 computation correct**: yellow=1.431, black=1.199

---

## BUG 15 [CRITICAL]: Coordinate Frame Mismatch — arm stalls at ~0.12m from target

### Evidence

Yellow carton GRASP trajectory (carton at [0.302, 0.700, 0.779]):
```
Step 150: eef=[0.201, 0.760, 0.905]  d=0.159  ← converging
Step 180: eef=[0.198, 0.752, 0.860]  d=0.131  ← Follow triggers here!
Step 210: eef=[0.196, 0.746, 0.836]  d=0.121  
Step 240: eef=[0.195, 0.744, 0.827]  d=0.119  ← stalling
Step 270: eef=[0.195, 0.744, 0.825]  d=0.119  ← STUCK
Step 420: eef=[0.193, 0.745, 0.825]  d=0.121  ← STILL STUCK
Step 432: Gripper closes at real_dist=0.126 (sub_step > 300 timeout)
```

Stall point [0.193, 0.745, 0.825] vs target [0.302, 0.700, 0.799]:
- **x error: 0.109m** (systematic, same for black carton: 0.504-0.393=0.111m)
- y error: 0.045m
- z error: 0.026m

Black carton APPROACH trajectory (carton at [0.504, 0.757, 0.779]):
```
Step  60: eef=[0.553, 0.568, 1.150]  d=0.295  ← moving correctly
Step  90: eef=[0.493, 0.658, 1.071]  d=0.173  ← moving, x passed target
Step 120: eef=[0.441, 0.727, 1.003]  d=0.102  ← x overshoots (0.441 < 0.504)
Step 150: eef=[0.409, 0.755, 0.969]  d=0.103  ← DIVERGING
Step 210: eef=[0.394, 0.761, 0.959]  d=0.114  ← STUCK
Step 240+: eef=[0.393, 0.761, 0.959]  d=0.114  ← FROZEN for 300 steps
```

### Root Cause: `target_local` and `obs["r_eef_pos"]` are in different frames

The function `_move_right_toward(target_l, obs["r_eef_pos"], ...)` computes:
```python
error = target_l - obs["r_eef_pos"]
```

But these are in **different coordinate frames**:
- `target_l` = `_corrected_world_to_arm(target_w, obs)` transforms through **real arm_base prim** (at [0.137, 0.093, 1.145])
- `obs["r_eef_pos"]` = IKFKSolver FK output, relative to **body FK arm_base** (at [0.565, 0.233, 1.060])

The body FK arm_base is **0.428m off in x, 0.140m off in y** from the real prim position. When these positions are subtracted, the resulting error vector doesn't represent the true displacement needed.

The arm initially moves in the right direction (because the error direction is approximately correct at large distances), but converges to the wrong position (because the absolute frame offset dominates at small distances). The ~0.11m systematic x-axis error is a direct consequence of this frame mismatch.

### Fix: Use relative error transform (rotation only)

Instead of transforming the absolute world target through T_inv (which produces coordinates in the real arm_base frame, incompatible with `obs["r_eef_pos"]`), compute the world-frame displacement from the real gripper to the target, rotate it to the arm_base local frame, and add it to `obs["r_eef_pos"]`:

```python
world_error = target_world - real_gripper_world     # vector in world frame
local_error = R_arm_base_inv @ world_error           # rotate to local frame
target_solver = obs["r_eef_pos"] + local_error       # target in solver's frame
```

This works because:
- `obs["r_eef_pos"]` is the current EEF in the solver's arm_base frame ✓
- `local_error` is the world displacement rotated to match the local frame orientation ✓
- The rotation from the real arm_base prim IS correct (same physical frame) ✓
- Only the position was wrong (body FK error), and we avoid using it ✓

---

## Implementation Plan

### File: `scripts/scripted_sorting_policy.py`

#### Change 1: New method `_world_target_to_solver_frame()`

Add this method right after `_corrected_arm_to_world` (around line 248):

```python
def _world_target_to_solver_frame(
    self, target_world: np.ndarray, obs: dict,
) -> np.ndarray:
    """Convert world target to IKFKSolver's local frame.

    Uses relative error: computes world displacement from real gripper
    to target, rotates to local frame, adds to solver's EEF position.
    This avoids the body-FK origin mismatch.
    """
    # Get real gripper world position
    real_grip = None
    if (self._shared_state is not None
            and self._shared_state.get("real_gripper_world")):
        real_grip = np.array(self._shared_state["real_gripper_world"])

    T = self._get_real_arm_base_T()

    if real_grip is not None and T is not None:
        # World-frame error vector
        world_error = target_world - real_grip
        # Rotate to arm_base local frame (T[:3,:3] maps local→world,
        # so its transpose maps world→local)
        R_inv = T[:3, :3].T
        local_error = R_inv @ world_error
        # Add to current solver EEF position
        return np.array(obs["r_eef_pos"]) + local_error

    # Fallback to old method
    return self._corrected_world_to_arm(target_world, obs)
```

#### Change 2: Update `_phase_approach` — use new method

Replace line 624:
```python
target_l = self._corrected_world_to_arm(target_w, obs)
```
With:
```python
target_l = self._world_target_to_solver_frame(target_w, obs)
```

#### Change 3: Update `_phase_grasp` — use new method in all 3 sub-phases

Replace line 658 (lowering):
```python
target_l = self._corrected_world_to_arm(target_w, obs)
```
With:
```python
target_l = self._world_target_to_solver_frame(target_w, obs)
```

Replace line 683 (lifting):
```python
target_l = self._corrected_world_to_arm(target_w, obs)
```
With:
```python
target_l = self._world_target_to_solver_frame(target_w, obs)
```

#### Change 4: Update `_phase_move_to_scanner` — use new method

Find all `_corrected_world_to_arm` calls in `_phase_move_to_scanner` and replace with `_world_target_to_solver_frame`.

#### Change 5: Update all remaining phases (`_phase_regrasp`, `_phase_move_to_bin`)

Find ALL remaining calls to `_corrected_world_to_arm` in phase handlers and replace with `_world_target_to_solver_frame`. The `_corrected_world_to_arm` method itself should be kept (not deleted) as a fallback inside the new method.

#### Change 6: Reduce APPROACH→GRASP transition threshold

Currently APPROACH transitions to GRASP after `sub_step > 500` (or `dist < 0.04`). With the frame fix, `dist` should decrease properly. Change:

```python
# In _phase_approach, line 632:
if dist < 0.04 or self.sub_step > 500:
```
To:
```python
if dist < 0.03 or self.sub_step > 300:
```

Also, the "holding near carton" branch (line 607-616) uses `world_dist < 0.2`. With the frame fix, the arm should move closer. Change:
```python
if world_dist < 0.2:
```
To:
```python
if world_dist < 0.10:
```
This prevents premature holding when the arm is still 0.15m away.

#### Change 7: Reduce gripper close distance threshold

Currently gripper closes at `real_dist < 0.06`. With the frame fix, the arm should get much closer. Add a diagnostic print when close:

```python
# In _phase_grasp, line 667:
if real_dist < 0.06 or dist < 0.03 or self.sub_step > 300:
```
To:
```python
if real_dist < 0.05 or dist < 0.02 or self.sub_step > 400:
    # Give more time for arm to converge, close at tighter distance
```

---

## Expected Behavior After Fix

1. **APPROACH**: arm_base local error correctly computed, arm converges in both x and y
   - Yellow carton: d_world should reach < 0.05m (vs 0.105m before)
   - Black carton: d_world should decrease steadily (vs FROZEN at 0.114m before)

2. **GRASP**: arm descends to within 0.05m of carton (vs stalling at 0.12m)
   - Gripper closes at real_dist < 0.05m (vs timeout at 0.126m)
   - PickUpOnGripper should trigger if carton is actually gripped

3. **Follow for black carton**: gripper should enter AABB (z range [0.679, 0.879])
   - Currently stuck at z=0.959, needs to descend to z < 0.879

---

## Verification Checklist

- [ ] Yellow carton: d_world during GRASP < 0.05m (was 0.119m)
- [ ] Yellow carton: PickUpOnGripper = 1 for at least 1 episode
- [ ] Black carton: Follow = 1 (gripper enters AABB)
- [ ] Black carton: arm not frozen during APPROACH
- [ ] No `[IK] Poor convergence` messages (ik_err < 0.12)
- [ ] `_world_target_to_solver_frame` used in all phase handlers

---

## Priority

1. **Changes 1–5**: Frame fix (fixes BUG 15 — the main reason arm stalls and PickUp fails)
2. **Change 6**: Tighter APPROACH threshold (prevents premature holding)
3. **Change 7**: Tighter gripper close distance
