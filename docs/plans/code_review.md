# Code Review & Modification Suggestions (Round 11)

**Date**: 2026-04-16
**Reviewer**: Local Claude Code (Evaluation)
**Target**: Cloud Claude Code reads this doc and modifies code

---

## Round 10 Result: Yellow PickUp works! Upright inconsistent. Average = 0.3542

### Scores (8 episodes)

| Episode | Carton | Follow | PickUp | Inside | Upright | PickUp2 | Inside2 |
|---------|--------|--------|--------|--------|---------|---------|---------|
| 1       | Yellow | **1** | **1** | **1** | **1** | 0 | 0 |
| 2       | Yellow | **1** | **1** | **1** | 0 | 0 | 0 |
| 3       | Yellow | **1** | **1** | **1** | 0 | 0 | 0 |
| 4       | Yellow | **1** | **1** | **1** | 0 | 0 | 0 |
| 5       | Black  | **1** | 0 | 0 | 0 | 0 | 0 |
| 6       | Black  | **1** | 0 | 0 | 0 | 0 | 0 |
| 7       | Black  | **1** | 0 | 0 | 0 | 0 | 0 |
| 8       | Black  | **1** | 0 | 0 | 0 | 0 | 0 |

**Average: 0.3542** (17/48 total points)

### What Works Now

1. **Yellow carton: Follow, PickUp, Inside all 4/4** — first 3 evaluators pass consistently
2. **Gripper closes at real_dist=0.034** — excellent grasp position
3. **PickUp detected at step ~180** — carton lifts successfully
4. **Inside triggers at step ~570** — carton enters scanner AABB during transport flyover
5. **Upright passed in Episode 1** — carton landed upright (angle 0.94°, threshold 5°)

---

## BUG 20 [CRITICAL]: MOVE_TO_SCANNER wastes 470 steps stalling, carton falls to ground

### Evidence

The MOVE_TO_SCANNER phase timeline (all 4 yellow episodes identical):

```
Sub 0-100:   Rotate waist to scanner (bj5 diff > 0.05)
Sub 100-200: Move above scanner — arm converges rapidly
             step 480 (sub 111): d_world=0.516
             step 510 (sub 141): d_world=0.361
             step 540 (sub 171): d_world=0.199
             step 570 (sub 201): d_world=0.074 → Inside triggers here!
             step 600 (sub 231): d_world=0.060 ← ARM STALLS

Sub 200-500: ARM STALLED at eef=[0.869, 0.001, 1.218], d_world=0.060
             *** 300 STEPS WASTED doing nothing ***

Sub 500-530: Lowering from z=1.218 to z=1.145 (30 steps)
Sub 530-700: ARM STALLED at eef=[0.874, 0.000, 1.145], d_world=0.055
             *** 170 STEPS WASTED doing nothing ***

Sub 700:     Gripper opens — carton drops
             Carton falls from z=1.112 to z=0.842 (HITS GROUND, not scanner!)
Sub 700-731: Hold open
Step 1100:   → REGRASP
```

**Total wasted steps: ~470 out of 731** (the arm is stalled 64% of the time!)

### Root Cause

The scanner at [0.929, 0.0, 1.163] is 0.060m beyond the arm's reach limit (x=0.869). The arm physically cannot reach the scanner center. Current code waits until `sub_step > 500` before lowering and `sub_step > 700` before releasing — burning hundreds of steps at the same stalled position.

When finally released, the carton is at x=0.874 (0.055m short of scanner center x=0.929). The carton falls 0.27m to the ground at z=0.842, missing the scanner surface entirely.

### Why Upright is inconsistent (1/4 yellow)

- Episode 1: Carton lands upright at z=0.842, Upright passes at step 1079 (31 steps before StepOut)
- Episodes 2-4: Carton tumbles slightly on landing, StepOut expires at step ~1109 before Upright passes

The Upright evaluator has ~929 steps budget (from PickUp at step 180 to StepOut at step ~1109). Currently MOVE_TO_SCANNER consumes 920 steps, leaving only ~9-30 steps for the carton to settle. This is too tight — any physics variation causes failure.

---

## BUG 21: REGRASP targets scanner position, but carton is on the ground

### Evidence

After release in Episode 1:
```
Carton actual position: [0.801, 0.028, 0.842]  ← on the ground!
REGRASP target:         [0.929, 0.000, 1.147]  ← scanner position (empty!)
```

The REGRASP phase tries to grab at the scanner, but the carton fell to the ground 0.31m away. The arm reaches d_world=0.056 from the scanner (same old stall point) and closes on air. PickUp2 always fails.

### Root Cause

`_phase_regrasp` always targets `self._scanner_pos`, ignoring where the carton actually is. The carton's real position is available via `_shared_state["real_carton_world"]`.

---

## BUG 22: Black carton unreachable (workspace limit)

### Evidence

Black carton at [0.504, 0.757, 0.779]:
```
Step 60:  eef=[0.574, 0.542, 1.119] d_world=0.360  (approaching)
Step 90:  eef=[0.535, 0.620, 1.003] d_world=0.217  (approaching)
Step 120: eef=[0.497, 0.678, 0.908] d_world=0.105  (approaching)
Step 150: eef=[0.478, 0.694, 0.875] d_world=0.077  ← Follow triggers, ARM STALLS
Step 180: eef=[0.475, 0.695, 0.872] d_world=0.076  ← no progress
Step 210: eef=[0.474, 0.695, 0.872] d_world=0.076  ← stalled permanently
```

Arm stalls 0.076m from target (real_dist to carton = 0.116m). Gripper never closes meaningfully. PickUp StepOut expires at step ~750.

### Root Cause

Horizontal distance from arm_base [0.137, 0.093] to carton [0.504, 0.757] = 0.759m. Arm maximum reach ≈ 0.69m. Deficit: 0.069m.

---

## Implementation Plan

### File: `scripts/scripted_sorting_policy.py`

#### Change 1 [HIGHEST PRIORITY]: Accelerate MOVE_TO_SCANNER — cut stall time

**Replace the entire `_phase_move_to_scanner` method** with a faster version that:
1. Reduces the "above_scanner" stall timeout from sub < 500 to sub < 280
2. Reduces the "lowering" stall timeout from sub < 700 to sub < 380
3. Adjusts the release and transition timing accordingly

```python
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

    # Sub 2: move above scanner (reduced timeout: 280 vs 500)
    target_w = self._scanner_pos.copy()
    target_w[2] += self.APPROACH_HEIGHT
    target_l = self._world_target_to_solver_frame(target_w, obs)
    new_joints, dist_above = self._move_right_toward(
        target_l, obs["r_eef_pos"], obs["r_eef_quat"],
        obs["arm_14"], step_size=self.EEF_STEP_FAST,
    )

    if dist_above > 0.05 and self.sub_step < 280:
        action = self._build_action(
            obs["left_arm"], new_joints, self._bj5_scanner, grip=1.0,
        )
        self._log(obs, target_w, "above_scanner")
        return action

    # Sub 3: lower onto scanner (reduced timeout: 380 vs 700)
    target_w = self._scanner_pos.copy()
    target_w[2] += self.GRASP_HEIGHT
    target_l = self._world_target_to_solver_frame(target_w, obs)
    new_joints, dist = self._move_right_toward(
        target_l, obs["r_eef_pos"], obs["r_eef_quat"],
        obs["arm_14"], step_size=self.EEF_STEP_SLOW,
    )

    if dist > 0.03 and self.sub_step < 380:
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
    if self.sub_step > 380 + self.RELEASE_HOLD_STEPS:
        self._set_phase("REGRASP")
    return action
```

**Expected timing improvement:**
- Old: MOVE_TO_SCANNER takes 731 sub-steps (step 369 → 1100)
- New: MOVE_TO_SCANNER takes ~440 sub-steps (step 369 → ~809)
- **Savings: ~290 steps** — gives the carton 290 more steps to settle for Upright

#### Change 2 [HIGH PRIORITY]: REGRASP uses actual carton position

The REGRASP phase must target where the carton actually is (on the ground), not where we wished it were (on the scanner). The carton's real position is available in `_shared_state["real_carton_world"]`.

**Replace the entire `_phase_regrasp` method:**

```python
def _phase_regrasp(self, obs: dict) -> np.ndarray:
    bj5_hold = self._bj5_scanner

    # Determine actual carton position for re-grasping
    regrasp_target = self._scanner_pos.copy()
    if (self._shared_state is not None
            and self._shared_state.get("real_carton_world")):
        regrasp_target = np.array(self._shared_state["real_carton_world"])
        # Update bj5 to face actual carton position
        bj5_hold = self._compute_bj5_for_target(regrasp_target)

    # Sub 1: lift hand (clear area above carton)
    if self.sub_step <= 50:
        target_w = regrasp_target.copy()
        target_w[2] += self.APPROACH_HEIGHT + 0.10  # higher clearance
        target_l = self._world_target_to_solver_frame(target_w, obs)
        new_bj5 = self._smooth_bj5(obs["bj5"], bj5_hold, self.BJ5_SPEED)
        new_joints, _ = self._move_right_toward(
            target_l, obs["r_eef_pos"], obs["r_eef_quat"],
            obs["arm_14"], step_size=self.EEF_STEP_FAST,
        )
        return self._build_action(obs["left_arm"], new_joints, new_bj5, grip=0.0)

    # Sub 2: wait for carton to settle
    if self.sub_step <= 80:
        new_bj5 = self._smooth_bj5(obs["bj5"], bj5_hold, self.BJ5_SPEED)
        return self._build_action(
            obs["left_arm"], self.last_right_arm, new_bj5, grip=0.0,
        )

    # Sub 3: approach above carton
    if self.sub_step <= 160:
        target_w = regrasp_target.copy()
        target_w[2] += self.APPROACH_HEIGHT
        target_l = self._world_target_to_solver_frame(target_w, obs)
        new_bj5 = self._smooth_bj5(obs["bj5"], bj5_hold, self.BJ5_SPEED)
        new_joints, _ = self._move_right_toward(
            target_l, obs["r_eef_pos"], obs["r_eef_quat"],
            obs["arm_14"], step_size=self.EEF_STEP_FAST,
        )
        action = self._build_action(obs["left_arm"], new_joints, new_bj5, grip=0.0)
        self._log(obs, target_w, "regrasp_above")
        return action

    # Sub 4: lower to carton
    if self.right_grip < 0.5:
        target_w = regrasp_target.copy()
        target_w[2] += self.GRASP_HEIGHT
        target_l = self._world_target_to_solver_frame(target_w, obs)
        new_joints, dist = self._move_right_toward(
            target_l, obs["r_eef_pos"], obs["r_eef_quat"],
            obs["arm_14"], step_size=self.EEF_STEP_SLOW,
        )
        action = self._build_action(obs["left_arm"], new_joints, bj5_hold, grip=0.0)
        self._log(obs, target_w, "regrasp_lower")

        # Check real distance to carton
        real_grip = None
        if (self._shared_state is not None
                and self._shared_state.get("real_gripper_world")):
            real_grip = np.array(self._shared_state["real_gripper_world"])
        real_dist = np.linalg.norm(real_grip - regrasp_target) if real_grip is not None else float('inf')

        if real_dist < 0.04 or dist < 0.02 or self.sub_step > 380:
            self.right_grip = 1.0
            self.sub_step = 380  # sync for hold counting
            print(f"[Policy] Re-grasp closing at step {self.step_count}"
                  f" real_dist={real_dist:.3f}")
        return action

    # Sub 5: hold
    if self.sub_step <= 380 + self.GRASP_HOLD_STEPS:
        return self._build_action(
            obs["left_arm"], self.last_right_arm, bj5_hold, grip=1.0,
        )

    # Sub 6: lift from ground
    target_w = regrasp_target.copy()
    target_w[2] += self.LIFT_HEIGHT
    target_l = self._world_target_to_solver_frame(target_w, obs)
    new_joints, dist = self._move_right_toward(
        target_l, obs["r_eef_pos"], obs["r_eef_quat"],
        obs["arm_14"], step_size=self.EEF_STEP_FAST,
    )
    action = self._build_action(obs["left_arm"], new_joints, bj5_hold, grip=1.0)
    self._log(obs, target_w, "regrasp_lift")

    if dist < 0.05 or self.sub_step > 380 + self.GRASP_HOLD_STEPS + 150:
        self._set_phase("MOVE_TO_BIN")
    return action
```

**Key differences from old REGRASP:**
- Uses `_shared_state["real_carton_world"]` for target position instead of `_scanner_pos`
- Computes bj5 for actual carton position (may differ from scanner bj5)
- Adds "approach above" phase before lowering (sub 80-160)
- Uses real gripper distance check for close threshold
- Tighter timeouts (380 vs 350 for close, shorter lift phase)

#### Change 3: Faster waist rotation speed

```python
BJ5_SPEED = 0.06          # rad/step for waist rotation (was 0.04)
```

This speeds up all waist rotations by 50%, saving ~33 steps per rotation (100→67 steps for a typical 60° rotation). Three rotations total (to scanner, to bin, return) saves ~100 steps overall.

#### Change 4: Reduce MOVE_TO_BIN timeouts

The MOVE_TO_BIN phase also has overly generous timeouts. Reduce them to save steps:

In `_phase_move_to_bin`, change:
```python
# Line: if abs(bj5 - self._bj5_bin) > 0.05 and self.sub_step < 150:
# Change 150 → 120 (rotation is faster with BJ5_SPEED=0.06)

# Line: if dist > 0.06 and self.sub_step < 400:
# Change 400 → 250

# Line: if self.sub_step > 400 + self.RELEASE_HOLD_STEPS:
# Change 400 → 250
```

---

## Expected Behavior After Fix

### Timeline comparison (yellow carton):

**Old (Round 10):**
```
Step   0-30:   INIT (30 steps)
Step  30-115:  APPROACH (85 steps)
Step 115-369:  GRASP (254 steps — close + hold + lift)
Step 369-1100: MOVE_TO_SCANNER (731 steps — 470 wasted!)
Step 1100+:    REGRASP (targeting scanner — carton not there)
               PickUp2 FAILS
```

**New (Round 11):**
```
Step   0-30:   INIT (30 steps)
Step  30-115:  APPROACH (85 steps)
Step 115-369:  GRASP (254 steps — same, works well)
Step 369-809:  MOVE_TO_SCANNER (440 steps — 290 saved!)
               Carton released at step ~749, settles by step ~780
               Upright checks carton at step ~780, ~330 steps before StepOut
Step 809-1200: REGRASP (targets actual carton on ground!)
               Carton at [0.801, 0.028, 0.842] — within arm reach (0.667m)
               Arm can reach, close, lift
               PickUp2 should trigger!
Step 1200-1450: MOVE_TO_BIN (faster rotations)
               Inside2 possible
```

### Expected score improvement:

| Metric | R10 (Yellow) | R11 Expected |
|--------|-------------|--------------|
| Follow | 4/4 | 4/4 |
| PickUp | 4/4 | 4/4 |
| Inside | 4/4 | 4/4 |
| Upright | 1/4 | 3-4/4 (290 more settling steps) |
| PickUp2 | 0/4 | 2-4/4 (targets real carton position) |
| Inside2 | 0/4 | 1-3/4 (if PickUp2 works, bin transport) |

**Projected score: 0.45-0.58** (vs 0.3542 current)

---

## Verification Checklist

- [ ] MOVE_TO_SCANNER completes in < 450 sub-steps (was 731)
- [ ] Carton has > 200 steps to settle after release (was < 30)
- [ ] Upright passes for >= 3/4 yellow episodes (was 1/4)
- [ ] REGRASP targets actual carton position (was scanner position)
- [ ] REGRASP logs show correct target coordinates
- [ ] PickUp2 triggers for >= 1 yellow episode (was 0/4)
- [ ] Waist rotations complete in < 70 steps (was ~100)
- [ ] No regression in Follow, PickUp, or Inside scores

---

## Priority Summary

1. **Change 1**: Cut MOVE_TO_SCANNER stall time (biggest impact — fixes Upright)
2. **Change 2**: REGRASP targets real carton position (enables PickUp2 + Inside2)
3. **Change 3**: Faster waist rotation (saves ~100 steps across all phases)
4. **Change 4**: Reduce MOVE_TO_BIN timeouts (minor savings)

---

## Not Addressed (Round 12+)

- **Black carton**: Requires body lean (bj2/bj4 adjustment) to extend reach by ~0.07m. Significant refactor of `_patched_step` body joint holding.
- **Scanner placement**: Arm can't reach scanner center (0.060m short). Would need body lean or different approach angle.
- **MOVE_TO_BIN accuracy**: Bin placement not yet verified — depends on PickUp2 working first.
