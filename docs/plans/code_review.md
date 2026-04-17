# Code Review & Modification Suggestions (Round 14)

**Date**: 2026-04-17
**Reviewer**: Local Claude Code (Evaluation)
**Target**: Cloud Claude Code reads this doc and modifies code

---

## Round 13 Result: MASSIVE PROGRESS — Score 0.5417 (vs Round 8 0.0833)

Commit `2a58d70` (Round 13 fixes: scanner place height + headless default).

### Score Breakdown (8 episodes, 6 sub-tasks each)

| Sub-task | Yellow (Ep1-4) | Black (Ep5-8) | Total |
|---|---|---|---|
| Follow | 1.0,1.0,1.0,1.0 | 1.0,1.0,1.0,1.0 | **8/8** ✅ |
| PickUpOnGripper (grasp) | 1,1,1,1 | 1,1,1,1 | **8/8** ✅ |
| Inside (scanner) | 1,1,1,1 | 1,1,1,1 | **8/8** ✅ |
| Upright | 0,0,0,**1** | 0,0,0,0 | 1/8 ❌ |
| PickUpOnGripper (regrasp) | 0,0,0,**1** | 0,0,0,0 | 1/8 ❌ |
| Inside (bin) | 0,0,0,0 | 0,0,0,0 | **0/8** ❌ |

**Average**: 0.5417 — **6.5x improvement** over Round 8 (0.0833), **+0.04 over Round 12 (0.5000)**

### What Works (massive wins from Rounds 9-13)

1. ✅ **Follow** — gripper follows target carton in AABB for ALL 8 episodes
2. ✅ **PickUpOnGripper** — successful grasp on first attempt for ALL 8
3. ✅ **Inside (scanner)** — carton placed inside scanner zone for ALL 8
4. ✅ Episode 4: full chain through REGRASP_PickUp succeeds (5/6 sub-tasks)

---

## BUG 24 [HIGH]: Carton not Upright on scanner (7/8 fail)

**Symptom**: After placing carton on scanner, the Upright check fails with `evt: 4` (timeout).
Only Episode 4 (yellow, by luck) achieves Upright with `dot product: 1.000, angle: 0.06°`.

**Evidence — Episode 4 SUCCESS log**:
```
Step 5377: [Upright] Object is upright: dot=1.000 (min: 0.996), angle: 0.06° (threshold: 5.00°)
```

**Evidence — Episode 1 FAILURE log**:
```
Step 1410: Action [Inside] evt: 3       ← carton enters scanner zone
Step 2107: Action [Upright] evt: 4      ← Upright TIMED OUT (never reached <5° tilt)
```

**Root cause**: The gripper holds the carton at whatever orientation the GRASP phase ended with.
During GRASP, IK converges joint configurations that may produce a tilted gripper.
When released onto scanner, the carton retains this tilt.

**Diagnostic data needed**: Log `obs["r_eef_quat"]` during MOVE_TO_SCANNER:
- At "above_scanner" sub-phase
- At release (gripper open) moment

Then convert to RPY to see how tilted the gripper is.

### Fix Strategy 24A: Force vertical gripper orientation during MOVE_TO_SCANNER

In `_phase_move_to_scanner()`, at "above_scanner" sub-phase, override the gripper rotation:

```python
# Force gripper to point straight down (z-axis vertical)
target_quat_world = np.array([0.0, 1.0, 0.0, 0.0])  # 180° flip about x
# Or compute from world yaw for the bj5 setting
```

Use `_compute_right_ik` with explicit target orientation that aligns the gripper's
"up" axis with world +z (so the carton's barcode-up face points to +z).

### Fix Strategy 24B: Add orientation correction step BEFORE release

Insert a new sub-phase between "above_scanner" and "lowering_to_scanner":
```
Sub 2.5 (re-orient): Rotate gripper joints so gripper local-z aligns with world -z.
```

This uses joint angles (esp. arm_r_link5/6/7) to physically rotate the wrist
without changing the gripper xy position. The carton inside the gripper rotates
with the gripper.

### Fix Strategy 24C [SIMPLEST]: Wait longer at scanner before transition

Maybe Episode 4 worked because it waited longer for carton to physically settle.
Increase MOVE_TO_SCANNER hold steps after release: from current value to 200+ steps.

But the failure log shows Upright `evt: 4` only fires at step 2107 (after 700+ steps),
so it's not just settling time. The carton is genuinely tilted on scanner.

---

## BUG 25 [HIGH]: Bin placement out of reach (8/8 fail)

**Symptom**: Episode 4 succeeded REGRASP_PickUp and reached MOVE_TO_BIN but
released the carton 0.385m short of the bin.

**Evidence — Episode 4 MOVE_TO_BIN trajectory**:
```
Bin world position: [0.300, -0.917, 0.838]
Step 1140: eef=[0.837,-0.096,1.503] tgt=[0.300,-0.917,0.838] d=1.185 [rotating_to_bin]
Step 1170: eef=[0.286,-0.293,1.673] tgt=[0.300,-0.917,0.838] d=1.042 [rotating_to_bin]
Step 1200: eef=[0.115,-0.361,1.646] tgt=[0.236,-0.521,0.988] d=0.688 [extending_to_bin]
Step 1290: eef=[0.199,-0.530,1.305] tgt=[0.236,-0.521,0.988] d=0.320 [extending_to_bin]
Step 1320: eef=[0.217,-0.532,1.148] tgt=[0.236,-0.521,0.988] d=0.162 [extending_to_bin]
                                                                      ↑ STALLED HERE
Step 1414: MOVE_TO_BIN → RETURN  ← phase ended without reaching bin
Step 6873: Action [Inside] evt: 4   ← carton landed OUTSIDE bin
```

The arm reached `[0.217, -0.532, 1.148]` but bin is at `[0.300, -0.917, 0.838]`:
- x error: -0.083m
- **y error: -0.385m** (bin is far in -y direction)
- z error: +0.310m (gripper too high)

**Root cause**: The bin at y=-0.917 is **outside** arm's reachable workspace
even with body lean. With arm_base at world y=0.093, the bin is 1.01m away in y alone.
The arm is dropping carton too high and too far from bin.

### Fix Strategy 25A: Use higher LIFT and rely on gravity drop

Drop the carton from a higher altitude with horizontal velocity toward bin.
Currently the carton is released at z=1.148m above ground (bin top at z=~0.84m,
fall distance = 0.31m). The carton needs both y velocity AND vertical fall.

Approach: at "extending_to_bin" with z held at ~1.3m, simply OPEN GRIPPER
(`right_grip = 0.0`). Carton drops vertically. Position must be directly above bin.

But arm can only reach y=-0.532 max, while bin is at y=-0.917. The release must
happen at the FURTHEST reachable point AND have body leaning toward bin.

### Fix Strategy 25B: Aggressive body lean toward bin

For MOVE_TO_BIN, override body joints to lean far toward -y direction:
- `bj1` (front-back lean): increase to extend reach
- `bj2`: similar to BJ2_LEAN logic from Round 12
- `bj3` (sideways lean): tilt toward bin

Example:
```python
# Lean body aggressively toward bin (-y direction)
BJ_BIN_LEAN = np.array([
    -1.5,   # bj1: lean further forward
    1.7,    # bj2: more forward extension
    -0.319, # bj3: unchanged
    0.3,    # bj4: small adjustment
    -1.517, # bj5: bin yaw
])
```

This may require physics testing — body joint limits could prevent this.

### Fix Strategy 25C: Use chassis movement (drive robot to bin)

The G2 robot has a wheeled chassis. The benchmark may allow chassis joint control.
Drive the robot toward -y by ~0.3m so the bin enters arm reach.

Check if `chassis_*` joints accept commands in the action space.

---

## BUG 26 [MEDIUM]: Premature MOVE_TO_BIN → RETURN transition

The MOVE_TO_BIN phase transitions to RETURN at sub_step=281 even though
d_world=0.162m to target. The condition for transition is too loose — carton
has not been released or has missed.

In `_phase_move_to_bin()`, ensure the carton is dropped INTO the bin before
phase ends. Use real-time carton z position: don't transition until carton has
dropped to z < 0.95 (bin top + small margin) AND xy is within bin AABB.

---

## Implementation Priority

1. **BUG 24** (Upright): Highest impact — fixing this unlocks 7 more PickUpOnGripper
   regrasps and 7 attempts at Inside(bin).
2. **BUG 25** (Bin reach): Need either body lean OR chassis movement.
3. **BUG 26** (Phase timing): Quick fix, prevents premature release.

---

## Verification Checklist

After fixes:
- [ ] `[Upright] Object is upright` log appears for ALL 8 episodes (not just Ep4)
- [ ] PickUpOnGripper (regrasp) score ≥ 6/8
- [ ] Inside (bin) evt: 3 (SUCCESS) for at least one episode
- [ ] Episode 4-style full chain success rate ≥ 4/8

---

## Round 13 Diagnostic Highlights

### Carton position tracking works perfectly
```
[Policy] Carton pos updated: [0.328, 0.687, 0.873] → [0.302, 0.700, 0.779] (diff=0.099m)
```
Real-time tracking confirmed — Round 8's BUG 13 fully resolved.

### Scanner placement works
```
Step 5277: MOVE_TO_SCANNER:142 eef=[0.996,0.024,1.292] [above_scanner]
Step 5377: [Upright] dot=1.000  ← ONLY Episode 4 reached vertical settle
```

### REGRASP picks up carton (when Upright succeeds)
```
Step 5844: [PickUpOnGripper] Grasp detected successfully
Step 6009: REGRASP → MOVE_TO_BIN  ← only Episode 4 reaches this
```

---

## Hotfixes Already Applied Locally

None this round — all changes from Round 13 (commit `2a58d70`) work as expected.

The remaining failures are fundamental control issues:
1. Gripper orientation not enforced during placement (BUG 24)
2. Bin out of arm reach (BUG 25)
