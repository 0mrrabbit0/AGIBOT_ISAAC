# Code Review & Modification Suggestions (Round 16)

**Date**: 2026-04-17
**Reviewer**: Local Claude Code (Evaluation)
**Target**: Cloud Claude Code reads this doc and modifies code

---

## Round 15 Result: NO CHANGE — Score 0.5000 (= Round 14, < Round 13's 0.5417)

Commit `bb888ee` (Round 15: revert world-down RPY + diagnostics + stabilize_above).

### Score Breakdown

| Sub-task | Yellow (Ep1-4) | Black (Ep5-8) | Total | Δ Round 14 |
|---|---|---|---|---|
| Follow | 1,1,1,1 | 1,1,1,1 | **8/8** ✅ | = |
| PickUpOnGripper (grasp) | 1,1,1,1 | 1,1,1,1 | **8/8** ✅ | = |
| Inside (scanner) | 1,1,1,1 | 1,1,1,1 | **8/8** ✅ | = |
| Upright | 0,0,0,0 | 0,0,0,0 | **0/8** ❌ | = |
| PickUpOnGripper (regrasp) | 0,0,0,0 | 0,0,0,0 | **0/8** ❌ | = |
| Inside (bin) | 0,0,0,0 | 0,0,0,0 | 0/8 ❌ | = |

**Average**: 0.5000 — same as Round 14. The revert worked (stable trajectory restored), but Upright still fails uniformly.

### Positive Outcomes

- ✅ MOVE_TO_SCANNER trajectory restored: eef ends at `[0.930, -0.000, 1.228]`, target `[0.929, 0.000, 1.227]` — perfect
- ✅ `stabilize_above` (sub 280-320) holds steady before descent — works as designed
- ✅ Scanner Inside check passes 8/8 (carton lands on scanner)
- ✅ REGRASP correctly tracks real carton position (e.g. tgt=`[0.873, 0.042, 0.906]` matches settled carton)
- ✅ Diagnostic logs deployed and captured

### Critical Diagnostic Result

**`GRASP-end gripper world_rpy=[1.413, 0.215, -2.360]`** (yellow) / `[1.470, 0.175, -2.622]` (black)

Roll ≈ **81–84°** from world +z, NOT ±180°. **The gripper is NOT pointing straight down at GRASP completion.** The carton is held tilted from the very start of the lift. No amount of post-GRASP correction can fix this without re-orienting mid-flight (risky).

---

## BUG 30 [CRITICAL]: Gripper not vertical at GRASP-end → carton tilted from pickup

**Symptom**: Across all 8 episodes, GRASP-end has identical orientation:
```
local_rpy = [0.303, -0.582, 2.673]   (arm_base frame)
world_rpy = [1.413, 0.215, -2.360]   (yellow) / [1.470, 0.175, -2.622] (black)
```

For a true "down-pointing" gripper, world_rpy should be `[±π, 0, yaw]` (gripper z-axis aligned with world -z). Current world roll of 1.413 rad means gripper z-axis is **only 81° from horizontal**, not 0°.

**Consequence**: The carton is grasped with its top face tilted ~99° from vertical. After release on scanner, the tilted carton tips over. Upright dot-product never exceeds 0.996 threshold → **0/8 across all episodes**.

### Settled-carton positions on scanner (Round 15)

| Episode | settled (x, y, z) | offset from scanner center [0.929, 0, 0.846] |
|---|---|---|
| Yellow Ep1 | [0.873, 0.042, 0.846] | (-0.056, +0.042) |
| Yellow Ep2 | [0.814, 0.042, 0.842] | (-0.115, +0.042) |
| Yellow Ep3 | [0.898, -0.006, 0.846] | (-0.031, -0.006) |
| Yellow Ep4 | [0.864, 0.057, 0.846] | (-0.065, +0.057) |
| Black Ep5  | [0.898, -0.089, 0.845] | (-0.031, -0.089) |
| Black Ep6  | [0.889, 0.055, 0.846] | (-0.040, +0.055) |
| Black Ep7  | [0.940, -0.004, 0.846] | (+0.011, -0.004) ← closest |
| Black Ep8  | [0.938, -0.008, 0.846] | (+0.009, -0.008) ← closest |

All settle at z=0.846 (scanner top, correct), but x/y wander by 0.03–0.12m — this is the **carton sliding/tipping after release** because it was held tilted.

### Fix Strategy 30A [PREFERRED]: Force vertical gripper DURING GRASP via target_rpy

The GRASP phase currently does pick-from-above motion via position-only IK. Add an orientation constraint that forces gripper z-axis to point world -z.

**Implementation steps**:

1. In `_phase_grasp`, after computing `target_pos` for GRASP sub-steps (especially "lower_to_carton" and "lift_carton"), compute target orientation in arm_base frame that maps gripper-z to world -z.

2. Use `_world_down_target_rpy()` (the function added in Round 14) — but apply it ONLY in GRASP, NOT in MOVE_TO_SCANNER. Key insight: in GRASP the arm is in a different waist/body pose where the IK constraint is feasible. In MOVE_TO_SCANNER the bj5 rotation makes the same constraint infeasible (caused R14 regression).

3. Validate the IK solution: if joint solution diverges by > 0.5 rad from previous step, fall back to position-only IK (don't break what works).

**Pseudocode** (add to `_phase_grasp`, in lower_to_carton and lift_carton sub-steps only):
```python
target_pos = ...  # existing position
target_rpy = self._world_down_target_rpy(obs)  # arm_base-frame RPY for world-down gripper
joints = self._ik_with_orientation(target_pos, target_rpy, obs, fallback_to_pos_only=True)
```

### Fix Strategy 30B [FALLBACK]: Pre-grasp wrist rotation

If 30A fails IK, before GRASP closes, rotate **bj7 (wrist roll)** alone to compensate. Read current world roll from r_eef_quat, compute delta to ±π, command wrist rotation. This is a single-DOF fix with no IK risk.

```python
# In _phase_grasp, after approach completes, before close gripper:
current_world_rpy = self._get_eef_rpy_world(obs["r_eef_quat"])
roll_error = math.pi - abs(current_world_rpy[0])  # how far from ±π
if abs(roll_error) > 0.1:
    arm_joints[6] += roll_error  # bj7 (or whichever joint controls wrist roll)
    # send action with adjusted wrist
```

### Fix Strategy 30C [SAFE BASELINE]: Lower scanner release height

Currently carton drops from z=1.198 to z=0.846 = **0.35m drop**. Even with vertical gripper, a 0.35m drop on a tilted object will tip it.

Reduce `SCANNER_PLACE_HEIGHT` so release happens closer to scanner surface:
```python
SCANNER_PLACE_HEIGHT = 0.02  # was 0.06, drop only 5cm
```

This minimizes tip-over even if orientation is imperfect. Combine with 30A/30B for best result.

---

## BUG 31 [HIGH]: REGRASP lifts carton successfully but PickUpOnGripper still scores 0

**Observation**: REGRASP trajectory shows the gripper IS lifting the carton:
```
[s=961]  real_grip=[0.874, 0.047, 0.877]  carton=[0.872, 0.050, 0.846]  dist=0.031  ← close
[s=991]  real_grip=[0.876, 0.057, 0.923]  carton=[0.874, 0.060, 0.892]  dist=0.031  ← carton z went up!
[s=1110] real_grip=[0.835, 0.053, 1.491]  carton=[0.845, 0.000, 1.461]  dist=0.030  ← carton at z=1.46
```

The carton is in the gripper and lifted by ~0.6m. Yet PickUpOnGripper scores 0.

**Hypothesis**: Action sequence `Action [StepOut] evt: 3` fires (Upright stage StepOut), advancing the action set BEFORE PickUpOnGripper for regrasp can register. The Upright failure cascades.

**Verification**: Once Upright passes (BUG 30 fix), check whether REGRASP's PickUpOnGripper auto-passes. If not, the StepOut for Upright is too short and needs investigation.

### Fix Strategy 31A: Wait for Upright fix first

Don't address this directly. Once carton lands upright, the action set should progress normally and recognize the regrasp lift.

---

## BUG 32 [MEDIUM]: Carton settles 0.03–0.12m off scanner center

Even with stabilize_above (which holds eef at scanner center for 5 sub-steps), the carton settles off-center. Two causes:

1. The gripper holds the carton ~0.03m below itself — but offset is x/y, not just z. The carton's CoM is shifted because it was grasped tilted (BUG 30).

2. Black episodes (5-8) have settled positions further from center because their GRASP-end orientation differs slightly (`world_rpy = [1.470, 0.175, -2.622]` vs `[1.413, 0.215, -2.360]`).

### Fix Strategy 32A: Compensate gripper-to-carton offset when targeting scanner

After GRASP, log the offset `carton_world - eef_world`. When commanding scanner placement, subtract this offset from target_pos so the **carton** (not the eef) lands on scanner center:
```python
# At end of GRASP:
self._gripper_carton_offset = carton_world - eef_world

# In MOVE_TO_SCANNER above_scanner sub-step:
target_eef = scanner_center - self._gripper_carton_offset
```

This works regardless of orientation — the eef positions itself such that the carton ends up centered.

---

## Implementation Priority for Round 16

1. **BUG 30A**: Add target_rpy=world-down ONLY in GRASP phase (lower_to_carton + lift_carton sub-steps). Use existing `_world_down_target_rpy()`. Test that GRASP still succeeds (8/8) — if any IK divergence, fall back to position-only.

2. **BUG 30C**: Reduce `SCANNER_PLACE_HEIGHT` from 0.06 to 0.02 (less drop = less chance to tip).

3. **BUG 32A**: Compute `_gripper_carton_offset` at GRASP-end, apply in MOVE_TO_SCANNER target.

4. Skip BUG 30B (wrist rotation) unless 30A is infeasible.

5. Keep all R15 changes (revert MOVE_TO_SCANNER world-down, stabilize_above, diagnostics).

---

## Verification Checklist

After Round 16 fixes:
- [ ] `[Diag] GRASP-end gripper world_rpy` shows roll near ±π (within 0.2 rad)
- [ ] PickUpOnGripper (grasp) still 8/8 (don't break what works)
- [ ] Inside (scanner) still 8/8
- [ ] **Upright ≥ 4/8** (clear improvement from 0/8)
- [ ] If Upright passes, REGRASP PickUpOnGripper should auto-improve

---

## Round 15 Diagnostic Highlights

### MOVE_TO_SCANNER trajectory (Yellow Ep1, all 8 are similar)
```
[s=390] eef=[0.561, 0.523, 1.540]  [rotating_to_scanner]   d=0.740
[s=420] eef=[0.804, 0.371, 1.478]  [rotating_to_scanner]   d=0.500
[s=450] eef=[0.918, 0.158, 1.431]  [above_scanner]         d=0.258
[s=480] eef=[0.983, 0.082, 1.362]  [above_scanner]         d=0.166
[s=510] eef=[0.998, 0.026, 1.295]  [above_scanner]         d=0.099
[s=540] eef=[0.931,-0.000, 1.228]  [stabilize_above]       d=0.002 ← REACHED
[s=570] eef=[0.930,-0.000, 1.228]  [stabilize_above]       d=0.001 ← HOLD
[s=600] eef=[0.930,-0.000, 1.228]  [stabilize_above]       d=0.001 ← HOLD
[s=630] eef=[0.930,-0.000, 1.228]  [stabilize_above]       d=0.001 ← HOLD
[s=660] eef=[0.930,-0.000, 1.228]  [stabilize_above]       d=0.001 ← HOLD
[carton released, falls to scanner]
[s=691] carton=[0.884, 0.019, 0.972]  ← falling
[s=721] carton=[0.873, 0.042, 0.846]  ← settled
```
The trajectory is excellent. The position is correct. **Only orientation is wrong.**

### REGRASP trajectory (Yellow Ep1)
```
[s=870]  REGRASP:91  eef=[0.873, 0.042, 0.955]  tgt=[0.873, 0.042, 0.906]  ← reach above
[s=900]  REGRASP:121 eef=[0.873, 0.042, 0.907]  tgt=[0.873, 0.042, 0.906]  ← descended
[s=930]  REGRASP:151 eef=[0.873, 0.042, 0.907]  tgt=[0.873, 0.042, 0.906]  ← hold
[s=961]  carton=[0.872, 0.050, 0.846]  dist=0.031  ← gripper near carton
[s=990]  REGRASP:423 eef=[0.876, 0.057, 0.923]  tgt=[0.873, 0.060, 1.182]  ← lifting!
[s=1110] REGRASP:543 eef=[0.835, 0.053, 1.491]  ← carton lifted to z=1.46
```
The lift visibly works. PickUpOnGripper failure is downstream of Upright failure.

### MOVE_TO_BIN trajectory (Yellow Ep1)
```
[s=1140] eef=[0.847,-0.008, 1.487]  tgt=[0.300,-0.917, 0.838]  [rotating_to_bin]
[s=1200] eef=[0.477,-0.328, 1.588]  tgt=[0.314,-0.634, 0.938]  [extending_to_bin]
[s=1290] eef=[0.393,-0.561, 1.237]
[s=1380] eef=[0.329,-0.622, 1.007]  ← reached extended target
[s=1500] eef=[0.317,-0.633, 0.976]  [releasing_into_bin]
```
Bin reach succeeded! Eef reaches `[0.317, -0.633, 0.976]`, target was `[0.314, -0.634, 0.938]` — within 4cm. But the carton was already lost (fake-grabbed during REGRASP because Upright stage was timed out).

---

## Recommendation Summary

**SAFEST PATH**:
1. Apply `_world_down_target_rpy` ONLY in GRASP phase (NOT scanner/bin) → fixes BUG 30
2. Reduce `SCANNER_PLACE_HEIGHT` to 0.02 → minimizes tip-over
3. Compute and apply `_gripper_carton_offset` in MOVE_TO_SCANNER → centers carton on scanner

**Expected outcome**: Upright should pass for at least 4/8 episodes, unlocking the cascade (REGRASP PickUpOnGripper, Bin Inside).

**Risk**: GRASP IK feasibility — verify with diagnostic. If 30A breaks GRASP, fall back to 30B (wrist-only rotation).
