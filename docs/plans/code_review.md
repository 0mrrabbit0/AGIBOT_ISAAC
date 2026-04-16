# Code Review & Modification Suggestions (Round 8)

**Date**: 2026-04-16
**Reviewer**: Local Claude Code (Evaluation)
**Target**: Cloud Claude Code reads this doc and modifies code

---

## Round 7 Result: 0 score, but MASSIVE progress

Commit `916c99c` + local hotfix (zero-quat guard, prim ordering).

### What Works Now

1. **`/genie/arm_base_link` found!** Real transform available:
   - pos=[0.1374, 0.0933, 1.1451]
   - rot=[-0.4949, 0.5054, 0.5050, -0.4945]
2. **bj5 computation correct!** bj5=1.431 for carton (was 3.048 before)
3. **APPROACH holds correctly** when d_world < 0.2 (gripper stays at [0.453, 0.600, 1.213])
4. **Arm converges toward target!** d_world: 0.191 → 0.078 → **0.037** in GRASP
5. **MOVE_TO_SCANNER waist rotation works!** d_world: 1.025 → 0.178

### Root Cause: Carton Position Stale After Physics Settle

**BUG 13 [CRITICAL]**: The carton position is queried at scene setup (before physics runs). But cartons FALL under gravity and settle at different z positions.

```
Setup query:  carton z = 1.097  (before physics)
Runtime step 1: carton z = 0.779  (after physics settle)
Height error:  0.318 m !!!
```

The arm moves to z≈1.097 (the stale target) while the carton is actually at z≈0.779. The arm is 0.32m above the carton. **Follow evaluator uses the carton's real-time position (z=0.779), so the gripper at z=1.14 is way outside the Follow AABB (z range [0.679, 0.879]).**

Trajectory proof — GRASP d_world is measured from STALE target, not real carton:
```
Step 210: eef_real=[0.309,0.672,1.144] tgt=[0.328,0.687,1.117] d_world=0.037 ← close to STALE target
                                       carton_actual=[0.302,0.700,0.779]       ← real carton 0.37m below!
```

---

## BUG 14: ARM_BASE_RPY offset may be incorrect

The code applies ARM_BASE_RPY (-π/2 around x) to the queried arm_base_link rotation. But the queried prim pose ALREADY includes the link's full orientation. IKFKSolver's arm_base frame may or may not include this RPY offset — we need to test both.

The arm converges but overshoots (d_world: 0.037 → 0.074 → 0.134 → 0.240), which could be caused by the RPY offset introducing a directional bias. However, the stale target position (BUG 13) is the primary issue — fix that first.

---

## Implementation Plan

### File 1: `scripts/run_sorting_benchmark.py`

#### Change 1A: Re-query carton position in `_patched_step` every step

The carton's real-time world position must be tracked, just like the gripper position. Add carton position tracking to `_patched_step`.

In the `_patched_step` closure variables (near `_diag_carton_name = target_carton_name`), add the carton prim path:

```python
_diag_carton_name = target_carton_name
_carton_prim_path = None
if target_carton_name:
    _carton_prim_path = f"/Workspace/Objects/{target_carton_name}"
```

In `_patched_step`, after the gripper query, add carton position query:

```python
# Query real carton position EVERY step
if _carton_prim_path:
    try:
        cp, _ = _env.api_core.get_obj_world_pose(_carton_prim_path)
        _shared_state["real_carton_world"] = [
            float(cp[0]), float(cp[1]), float(cp[2])
        ]
    except Exception:
        pass
```

#### Change 1B: Fix zero-quaternion guard in arm_base candidates

The current candidate list starts with `/genie/arm_r_base_link` which returns a zero quaternion. Fix:

```python
for cand in [
    "/genie/arm_base_link",       # correct prim name (from enum)
    "/genie/arm_r_base_link",
    "/genie/arm_r_link1",
]:
    try:
        ab_p, ab_r = _env.api_core.get_obj_world_pose(cand)
        # Skip zero quaternions (invalid prim)
        import math
        qnorm = math.sqrt(sum(
            float(ab_r[i]) ** 2 for i in range(4)
        ))
        if qnorm < 0.01:
            print(f"[ArmBase] {cand}: zero quat, skip")
            continue
        # ... rest of existing code
```

(This was already hotfixed locally but needs to be committed.)

### File 2: `scripts/scripted_sorting_policy.py`

#### Change 2A: Use real-time carton position from shared state

In `act()`, before dispatching to phase handlers, update `_carton_pos` from shared state:

```python
# Update carton position from real-time simulation query
if (self._shared_state is not None
        and self._shared_state.get("real_carton_world")):
    new_pos = np.array(self._shared_state["real_carton_world"])
    if self._carton_pos is not None:
        # Only update if significantly different (carton has settled)
        diff = np.linalg.norm(new_pos - self._carton_pos)
        if diff > 0.01:
            if self.step_count <= 5:
                print(f"[Policy] Carton pos updated: "
                      f"{self._carton_pos} → {new_pos} "
                      f"(diff={diff:.3f}m)")
            self._carton_pos = new_pos
    else:
        self._carton_pos = new_pos
```

Place this BEFORE the fallback positions block and BEFORE the phase handler dispatch.

#### Change 2B: Add zero-quaternion guard in `_get_real_arm_base_T`

```python
pos = np.array(ab_pos)
qw, qx, qy, qz = ab_rot
# Guard against zero quaternion
qnorm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
if qnorm < 0.01:
    return None
```

(Also hotfixed locally, needs committing.)

#### Change 2C: Try WITHOUT ARM_BASE_RPY offset

The queried `/genie/arm_base_link` world pose already includes the link orientation. The RPY offset may not be needed. **Comment out the RPY rotation** to test:

```python
def _get_real_arm_base_T(self) -> np.ndarray | None:
    # ... existing code ...
    if R is not None:
        rot_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
    else:
        rot_mat = np.eye(3)

    # NOTE: Do NOT apply ARM_BASE_RPY — the queried prim pose
    # already includes the link's full orientation. Applying
    # an extra -pi/2 rotation causes directional errors.
    # rpy_rot = self._rot('x', self.ARM_BASE_RPY[0])
    # rot_mat = rot_mat @ rpy_rot

    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos
    return T
```

**If results get WORSE** (arm moves in completely wrong direction), then the RPY IS needed. In that case, restore it. But the overshoot pattern suggests it should be removed.

#### Change 2D: Fix GRASP gripper close timing — use real-time distance

Currently GRASP closes the gripper when FK-computed d < 0.03 or sub_step > 300. With real-time carton position, use world-frame distance:

```python
def _phase_grasp(self, obs):
    bj5_hold = obs["bj5"]

    # Get real-time distance to carton
    real_grip = None
    if (self._shared_state is not None
            and self._shared_state.get("real_gripper_world")):
        real_grip = np.array(self._shared_state["real_gripper_world"])
    if real_grip is not None:
        real_dist = np.linalg.norm(real_grip - self._carton_pos)
    else:
        real_dist = float('inf')

    # Sub 1: lower to carton
    if self.right_grip < 0.5:
        target_w = self._carton_pos.copy()
        target_w[2] += self.GRASP_HEIGHT
        target_l = self._corrected_world_to_arm(target_w, obs)
        new_joints, dist = self._move_right_toward(
            target_l, obs["r_eef_pos"], obs["r_eef_quat"],
            obs["arm_14"], step_size=self.EEF_STEP_SLOW,
        )
        action = self._build_action(obs["left_arm"], new_joints, bj5_hold)
        self._log(obs, target_w, "lowering")

        # Close gripper when real distance is small enough
        if real_dist < 0.06 or dist < 0.03 or self.sub_step > 300:
            self.right_grip = 1.0
            self.sub_step = 0
            print(f"[Policy] Gripper closing at step {self.step_count}"
                  f" real_dist={real_dist:.3f}")
        return action

    # ... rest of existing code unchanged ...
```

---

## Expected Behavior After Fix

1. At step 1, `_carton_pos` updates from [0.328, 0.687, 1.097] to [0.302, 0.700, 0.779]
2. APPROACH: gripper at [0.453, 0.600, 1.213], carton at z≈0.779
   - d_world = √((0.453-0.302)² + (0.600-0.700)² + (1.213-0.779)²) ≈ 0.46m
   - NOT within 0.2m, so arm will try to MOVE instead of holding
3. Arm moves toward [0.302, 0.700, 0.779+0.15] = [0.302, 0.700, 0.929]
4. As arm descends, gripper enters carton AABB → Follow triggers!
5. GRASP: gripper closes when real_dist < 0.06

---

## Verification Checklist

- [ ] `[Policy] Carton pos updated: [old] → [new]` appears at step 1
- [ ] GRASP d_world measured against actual carton position (z≈0.78)
- [ ] Gripper enters carton AABB (z within [0.679, 0.879])
- [ ] Follow score > 0 for at least one episode
- [ ] `[ArmBase] FOUND: /genie/arm_base_link` (not arm_r_base_link)

---

## Priority

1. **Change 1A + 2A**: Real-time carton position (fixes BUG 13 — the main reason Follow fails)
2. **Change 1B + 2B**: Zero-quaternion guard (prevents crash)
3. **Change 2C**: Remove ARM_BASE_RPY offset (fixes arm overshoot)
4. **Change 2D**: GRASP close timing with real-time distance

---

## Hotfixes Already Applied Locally (need committing)

The following changes were made locally to fix the zero-quaternion crash:
1. `run_sorting_benchmark.py`: Reordered arm_base candidates, added qnorm check
2. `scripted_sorting_policy.py`: Added qnorm < 0.01 guard in `_get_real_arm_base_T`

These should be committed along with the new changes above.
