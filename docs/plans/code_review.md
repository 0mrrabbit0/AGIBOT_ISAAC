# Code Review & Modification Suggestions (Round 15)

**Date**: 2026-04-17
**Reviewer**: Local Claude Code (Evaluation)
**Target**: Cloud Claude Code reads this doc and modifies code

---

## Round 14 Result: REGRESSION — Score 0.5000 (vs Round 13's 0.5417)

Commit `8a8cb8c` (Round 14 fixes: world-down gripper + extended bin reach).

### Score Breakdown

| Sub-task | Yellow (Ep1-4) | Black (Ep5-8) | Total | Δ Round 13 |
|---|---|---|---|---|
| Follow | 1,1,1,1 | 1,1,1,1 | **8/8** ✅ | = |
| PickUpOnGripper (grasp) | 1,1,1,1 | 1,1,1,1 | **8/8** ✅ | = |
| Inside (scanner) | 1,1,1,1 | 1,1,1,1 | **8/8** ✅ | = |
| Upright | 0,0,0,0 | 0,0,0,0 | **0/8** ❌ | **−1** |
| PickUpOnGripper (regrasp) | 0,0,0,0 | 0,0,0,0 | **0/8** ❌ | **−1** |
| Inside (bin) | 0,0,0,0 | 0,0,0,0 | 0/8 ❌ | = |

**Average**: 0.5000 — **regressed −0.04** from Round 13.

The world-down RPY change destroyed Episode 4's lucky Upright success without
helping any other episode. **The fix made things worse.**

---

## BUG 27 [CRITICAL]: world-down RPY override moves arm in WRONG direction

**Symptom**: With `_world_down_target_rpy` applied during MOVE_TO_SCANNER, the
arm overshoots the scanner in the -y direction.

**Trajectory comparison**:

Round 13 (works, no world-down):
```
MOVE_TO_SCANNER:480 eef=[0.985, 0.079, 1.359] tgt=[0.929, 0.000, 1.227]  ← above scanner
MOVE_TO_SCANNER:510 eef=[0.997, 0.026, 1.294]                            ← descending
```

Round 14 (broken, with world-down):
```
MOVE_TO_SCANNER:480 eef=[0.806, -0.117, 1.219]   ← already y=-0.117 (scanner y=0)
MOVE_TO_SCANNER:660 eef=[0.838, -0.192, 1.096]   ← lowering_scanner started, off-target
MOVE_TO_SCANNER:720 eef=[0.850, -0.233, 1.074]   ← ended at y=-0.233 (0.23m past scanner!)
```

The arm released the carton at [0.85, -0.23, 1.07] — 0.23m off from scanner center
in -y direction. Inside check still fires (table is wide), but carton lands tilted/off-center.

**REGRASP also broken**:

Round 13 REGRASP:91 eef=[0.875, -0.030, 0.910]  ← above carton on scanner
Round 14 REGRASP:91 eef=[0.271, -0.690, 0.815]  ← CRAZY position (y=-0.690!)

The world-down RPY caused the IK to choose drastically different joint configurations
that move the gripper to entirely wrong xy locations.

### Root cause analysis

`_world_down_target_rpy` likely computes a target RPY in the arm_base local frame
that, when commanded, conflicts with position targeting. Two issues:

1. **Over-constraint**: Forcing both position AND a specific orientation may push
   IK into singular configurations or cause the solver to prioritize orientation
   over position.

2. **Incorrect transform**: The conversion from world-down to local-frame RPY may
   have a sign error or axis swap. The `arm_base_link` rotation includes a 90°
   transform; the world -z direction maps to a non-trivial local axis.

### Fix Strategy 27A: REVERT the world-down RPY override

Restore Round 13's behavior where MOVE_TO_SCANNER uses the GRASP-time gripper
orientation (whatever it was). Episode 4 succeeded by luck in Round 13, so
average baseline can be recovered.

### Fix Strategy 27B: Re-derive _world_down_target_rpy with verification

If keeping the world-down concept:
1. Log `obs["r_eef_quat"]` BEFORE and AFTER the override at first call
2. Verify the resulting world-frame orientation IS pointing -z
3. Test in isolation with no body lean first

---

## BUG 28 [HIGH]: Carton settled OUTSIDE bbox center on scanner

Even with Inside=1, the carton's settled position is far from scanner center.
This causes Upright to fail because the carton lands on the scanner's edge or
support struts.

**Diagnostic data needed**:
After scanner placement, log the carton's settled (xyz, rpy):
```python
# In _phase_move_to_scanner, after release sub-step:
real_carton = self._shared_state.get("real_carton_world")
print(f"[Diag] Carton settled at {real_carton}")
```

If carton is off-center, the next REGRASP needs to query its real position
(not the original scanner position) before attempting to grasp.

Round 13 already does this for REGRASP → real carton tracking works. But the
carton lands tilted regardless of position because of orientation issues.

### Fix Strategy 28A: Approach scanner from above, descend straight down

Currently MOVE_TO_SCANNER does "rotating_to_scanner" → "above_scanner" → "lowering".
The lowering part may be sliding the carton off-center.

Modification:
1. Reach a high position DIRECTLY above scanner: target = (scanner_xy, scanner_z + 0.20)
2. Hold position for 30 steps to stabilize
3. Lower vertically (only z changes) to scanner_z + SCANNER_PLACE_HEIGHT
4. Open gripper

This decouples xy positioning from vertical descent.

---

## BUG 29 [HIGH]: Upright fundamentally needs gripper to release in vertical pose

The carton is held by the gripper at whatever orientation the gripper has.
For the carton to land upright, its z-axis (the "up" face in carton frame)
must be parallel to world +z.

**Cleanest approach**: At INIT, the GRASP picks the carton from above with the
gripper pointing down. If the gripper STAYS pointing down through MOVE_TO_SCANNER,
the carton stays upright.

**Why it fails**: The bj5 (waist) rotation during MOVE_TO_SCANNER rotates the
ENTIRE arm including the gripper. If the gripper was vertical before bj5 rotation,
it remains vertical after (pure yaw rotation preserves vertical alignment).

**Hypothesis**: Maybe the GRASP gripper is NOT vertical to start with. Then no
amount of waist rotation will make it vertical at scanner.

### Fix Strategy 29A: Diagnostic — log gripper RPY at GRASP completion

```python
# At end of GRASP phase (after lift):
rpy = self._get_eef_rpy(obs["r_eef_quat"])
print(f"[Diag] Gripper RPY at GRASP-end: roll={rpy[0]:.3f} pitch={rpy[1]:.3f} yaw={rpy[2]:.3f}")
```

Expected: roll≈±π (gripper flipped 180° to point down), pitch≈0.

If gripper is not pointing down at GRASP end, fix GRASP to use vertical orientation.

### Fix Strategy 29B: Force vertical gripper at GRASP, not at MOVE_TO_SCANNER

Apply orientation control DURING GRASP (when picking up carton) rather than after.
The gripper should point straight down (-z in world) when closing on the carton.

This way the carton is held vertically from the start.

---

## Implementation Priority for Round 15

1. **BUG 27**: REVERT world-down RPY in MOVE_TO_SCANNER (restore Round 13 behavior).
   Recovery to 0.5417 baseline.

2. **BUG 29A**: Add diagnostic logs for gripper RPY at GRASP-end and at scanner
   release. This data is essential before attempting orientation fixes.

3. **BUG 28A**: Decouple xy positioning from vertical descent at scanner.

4. Skip world-down for MOVE_TO_BIN until BUG 27 is resolved (same root cause).

---

## Verification Checklist

After Round 15 fixes:
- [ ] Average score ≥ 0.5417 (recover Round 13 baseline)
- [ ] `[Diag] Gripper RPY at GRASP-end` log appears
- [ ] MOVE_TO_SCANNER trajectory ends with eef.y near 0.0 (not -0.2+)
- [ ] At least 1 episode achieves Upright again

---

## Round 14 Diagnostic Highlights

### MOVE_TO_SCANNER trajectory (yellow, Ep1)
```
[s=480] MOVE_TO_SCANNER:112 eef_real=[0.806,-0.117,1.219] [above_scanner] ← already off
[s=510] MOVE_TO_SCANNER:142 eef_real=[0.833,-0.156,1.142] [above_scanner]
[s=540] MOVE_TO_SCANNER:172 eef_real=[0.839,-0.170,1.119] [above_scanner]
[s=660] MOVE_TO_SCANNER:292 eef_real=[0.838,-0.192,1.096] [lowering_scanner]
[s=720] MOVE_TO_SCANNER:352 eef_real=[0.850,-0.233,1.074] [lowering_scanner] ← ENDS HERE
```
The arm continues moving in -y throughout. Target was [0.929, 0.000, 1.227].

### REGRASP at wrong position
```
[s=870] REGRASP:91 eef_real=[0.271,-0.690,0.815] tgt=[0.926,-0.300,0.838]
```
The arm starts REGRASP at y=-0.69 — completely outside the scanner area.

### Episode 4 in Round 13 (lucky success) NOT replicated in Round 14
The world-down RPY change altered IK joint config across all episodes,
breaking the lucky alignment that made Ep4 succeed.

---

## Recommendation Summary

**SAFEST PATH**: Revert Round 14's `_world_down_target_rpy` calls in BOTH
MOVE_TO_SCANNER and MOVE_TO_BIN. Recover 0.5417 baseline. Then add diagnostic
logs to understand actual gripper orientation before attempting orientation control.

**Risky changes that need verification BEFORE deployment**:
- Any IK orientation override
- Body joint changes (BJ_BIN_LEAN was untested as no episode reached MOVE_TO_BIN extending)
