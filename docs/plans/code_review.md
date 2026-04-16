# Code Review & Modification Suggestions (Round 10)

**Date**: 2026-04-16
**Reviewer**: Local Claude Code (Evaluation)
**Target**: Cloud Claude Code reads this doc and modifies code

---

## Round 9 Result: Follow = 1.0 for ALL 8 episodes! Average = 0.1667

### Scores (8 episodes)

| Episode | Carton | Follow | PickUp | Inside | Upright | PickUp2 | Inside2 |
|---------|--------|--------|--------|--------|---------|---------|---------|
| 1–4     | Yellow | **1.0** | 0 | 0 | 0 | 0 | 0 |
| 5–8     | Black  | **1.0** | 0 | 0 | 0 | 0 | 0 |

**Average: 0.1667** (8 Follow successes / 48 total points)

### What Works Now

1. **Frame mismatch fix works!** `_world_target_to_solver_frame` eliminated the systematic 0.11m x-error
2. **Yellow carton**: APPROACH converges to d=0.085 in 60 steps (was 0.105 in R8)
3. **Yellow carton**: GRASP converges to d=0.055 in 47 steps (was 0.119 in R8)
4. **Yellow carton**: Gripper closes at real_dist=0.049 (was 0.126 timeout in R8)
5. **Black carton**: Follow now triggers! Arm reaches [0.473, 0.692, 0.869] — inside AABB
6. **Black carton**: d_world reduced from 0.114 (R8, completely frozen) to 0.096 (R9)

---

## BUG 18 [CRITICAL]: Gripper closes too high — carton lifts 0.004m then drops

### Evidence

Yellow carton GRASP sequence:
```
Step 150 (GRASP:47): eef=[0.302, 0.682, 0.851] d=0.055  ← Follow triggers
Step 166 (GRASP:63): Gripper closes at real_dist=0.049
Step 181 (after close): carton z = 0.783 (was 0.779, lifted 0.004m!)
Step 211 (lifting):     carton z = 0.779 (dropped back — LOST GRIP)
```

The gripper at z=0.851 is at the very TOP of the carton (carton center z=0.779, carton half-height ~0.075m, top ≈ 0.854). The fingers barely catch the edge, lift 0.004m, then the carton slips out.

**PickUpOnGripper requires lift > 0.02m** — we achieve only 0.004m.

### Root Cause: Arm can't descend from z=0.851 to z=0.799 (target)

The GRASP descent rate:
```
GRASP:17 → z=0.911  (start)
GRASP:47 → z=0.851  (0.060m in 30 steps = 0.002m/step)
GRASP:63 → z≈0.841  (0.010m in 16 steps = 0.0006m/step ← decelerating)
```

The arm is decelerating as it approaches a joint configuration limit. At z=0.851, the gripper close threshold (real_dist < 0.05) triggers, closing the gripper prematurely while still 0.072m above the carton center.

---

## BUG 19: Black carton at workspace boundary (d=0.096, arm stalls)

The black carton at [0.504, 0.757, 0.779] is 0.759m from arm_base [0.137, 0.093, 1.145] in the x-y plane — near the arm's workspace limit (~0.75-0.8m reach).

The arm stalls at [0.467, 0.681, 0.845]:
- x: 0.037m short
- y: 0.076m short
- z: 0.046m high

The gripper never closes (real_dist=0.107, sub_step only reaches 389 before StepOut at step 720).

---

## Implementation Plan

### File: `scripts/scripted_sorting_policy.py`

#### Change 1: Lower APPROACH_HEIGHT from 0.15 to 0.06

The arm starts GRASP from z = carton_z + APPROACH_HEIGHT. At 0.15m, the arm starts GRASP at z≈0.929 and must descend 0.130m. At 0.06m, it starts at z≈0.839 and only needs to descend 0.040m — well within the convergence capability.

```python
APPROACH_HEIGHT = 0.06    # m above target for pre-approach (was 0.15)
```

This is the **most impactful single change** — it puts the arm closer to grasp height when GRASP begins.

#### Change 2: Lower GRASP_HEIGHT from 0.02 to -0.02

Target the gripper 0.02m BELOW carton center. This ensures the fingers close around the carton body, not the top edge:

```python
GRASP_HEIGHT = -0.02      # m relative to carton center (was 0.02)
```

At carton z=0.779, target becomes z=0.759 — firmly in the middle of the carton.

#### Change 3: Tighten gripper close threshold

Don't close the gripper until the gripper is truly close to the carton:

```python
# In _phase_grasp, change the close condition:
if real_dist < 0.035 or dist < 0.015 or self.sub_step > 500:
```

Was: `real_dist < 0.05 or dist < 0.02 or self.sub_step > 400`

This gives the arm more time to descend and ensures the gripper closes at z closer to the carton center.

#### Change 4: Increase EEF_STEP_SLOW for faster GRASP descent

```python
EEF_STEP_SLOW = 0.012     # m/step for precise moves (was 0.008)
```

This makes the arm descend faster during GRASP, helping it reach the target before stalling.

#### Change 5: Increase IK iterations

```python
# In _compute_right_ik, change n_iter default:
def _compute_right_ik(
    self,
    target_pos: np.ndarray,
    target_rpy: np.ndarray,
    current_arm_14: np.ndarray,
    n_iter: int = 15,        # was 10
) -> np.ndarray | None:
```

More iterations help the IK solver find better solutions near workspace boundaries.

#### Change 6: Increase LIFT_HEIGHT for stronger PickUp signal

```python
LIFT_HEIGHT = 0.30         # m above target after grasping (was 0.22)
```

Higher lift ensures PickUpOnGripper detects the lift (> 0.02m above initial position). Also keeps the carton clear when rotating to scanner.

#### Change 7: Reduce APPROACH hold threshold

```python
# In _phase_approach, line ~628:
if world_dist < 0.06:    # was 0.10
```

With the lower APPROACH_HEIGHT (0.06m), the gripper at carton_z + 0.06 is closer to the carton. Only hold if world_dist < 0.06m to prevent premature holding.

---

## Expected Behavior After Fix

### Yellow carton:
1. APPROACH target: z = 0.779 + 0.06 = 0.839 (was 0.929)
2. Arm reaches z≈0.839 during APPROACH (vs z≈0.929 before)
3. GRASP target: z = 0.779 - 0.02 = 0.759 (was 0.799)
4. Arm descends from 0.839 to ≈0.78 (only 0.059m descent needed!)
5. Gripper closes at real_dist < 0.035, with z ≈ 0.78 — IN the carton body
6. Carton lifts > 0.02m → PickUpOnGripper = 1

### Black carton:
1. APPROACH target: z = 0.779 + 0.06 = 0.839
2. Arm still limited by workspace (0.759m reach)
3. But lower z target reduces the reach distance slightly
4. With EEF_STEP_SLOW=0.012, arm converges faster
5. May reach close enough for gripper to close

---

## Verification Checklist

- [ ] Yellow: Gripper closes at z < 0.82 (was 0.851)
- [ ] Yellow: Carton lifts > 0.02m after grip (was 0.004m)
- [ ] Yellow: PickUpOnGripper = 1 for at least 1 episode
- [ ] Black: d_world < 0.08 during GRASP (was 0.096)
- [ ] APPROACH_HEIGHT change doesn't break Follow (gripper still enters AABB)
- [ ] GRASP_HEIGHT negative doesn't cause collision issues

---

## Priority

1. **Change 1**: Lower APPROACH_HEIGHT (most impactful — reduces descent needed)
2. **Change 2**: Lower GRASP_HEIGHT (targets carton center, not top)
3. **Change 3**: Tighten gripper close threshold (close only when properly positioned)
4. **Change 4**: Faster GRASP step size
5. **Changes 5-7**: Supporting improvements

---

## Key Insight

The arm CAN reach [0.302, 0.682, 0.851] for the yellow carton — that's only 0.072m above the carton center. With APPROACH_HEIGHT=0.06 (arm starts closer to carton height) and GRASP_HEIGHT=-0.02 (target below carton center), the arm only needs to descend 0.04m during GRASP instead of 0.13m. This is well within the arm's convergence capability, which shows ~0.06m descent in 30 steps.
