# Code Review & Modification Suggestions (Round 13)

**Date**: 2026-04-17
**Reviewer**: Local Claude Code (Evaluation)
**Target**: Cloud Claude Code reads this doc and modifies code

---

## Round 12 Result: Body lean works! Black carton PickUp solved. Average = 0.5000

### Scores (8 episodes)

| Episode | Carton | Follow | PickUp | Inside | Upright | PickUp2 | Inside2 |
|---------|--------|--------|--------|--------|---------|---------|---------|
| 1       | Yellow | **1** | **1** | **1** | 0 | 0 | 0 |
| 2       | Yellow | **1** | **1** | **1** | 0 | 0 | 0 |
| 3       | Yellow | **1** | **1** | **1** | 0 | 0 | 0 |
| 4       | Yellow | **1** | **1** | **1** | 0 | 0 | 0 |
| 5       | Black  | **1** | **1** | **1** | 0 | 0 | 0 |
| 6       | Black  | **1** | **1** | **1** | 0 | 0 | 0 |
| 7       | Black  | **1** | **1** | **1** | 0 | 0 | 0 |
| 8       | Black  | **1** | **1** | **1** | 0 | 0 | 0 |

**Average: 0.5000** (24/48 total points) — up from 0.3333

### What Works Now

1. **Body lean (bj2 +0.27 rad) works perfectly** — arm reaches x=0.997 (was x=0.874)
2. **Black carton PickUp+Inside now 4/4** — was 0/4 in R11. Body lean fully solved the reach problem
3. **All 8 episodes: Follow+PickUp+Inside pass** — 24/48 points
4. **REGRASP successfully re-grips** — real_dist=0.037 at close, carton lifts

### Score progression

R8=0.083 → R9=0.167 → R10=0.354 → R11=0.333 → **R12=0.500**

---

## BUG 23 [CRITICAL]: Carton released too high — falls 0.30m to ground, Upright fails

### Evidence (all 8 episodes identical pattern)

```
Step 570 (sub=202): eef=[0.932,-0.001,1.179] carton=[0.931,0.005,1.144] d_world=0.032 [lowering_scanner]
Step 600 (sub=232): eef=[0.930,-0.001,1.180] carton=[0.930,0.004,1.145] d_world=0.032 [lowering_scanner]
Step 630 (sub=262): eef=[0.930,-0.001,1.181] carton=[0.929,0.004,1.145] d_world=0.033 [lowering_scanner]

*** Between step 631-661: gripper opens, carton drops ***

Step 661: carton=[0.879,0.011,0.846]  ← fell 0.299m to ground!
Step 779 (sub=411): MOVE_TO_SCANNER → REGRASP
```

### Root Cause

The lowering target is `scanner_pos[2] + GRASP_HEIGHT = 1.167 + (-0.02) = 1.147`. The arm reaches z=1.179 (0.032m above target) and stalls — IK workspace limit prevents going lower.

When `dist` fluctuates below 0.03 (the `if dist > 0.03` threshold), the code falls through to the release phase. The gripper opens. The carton at z=1.145 is 0.022m BELOW the scanner center (z=1.167). The carton is NOT on the scanner surface — it's hanging in mid-air from the gripper. When released, it falls 0.299m to the ground.

**Why the carton doesn't land on the scanner:**
The gripper-to-carton offset is 0.034m (carton center is 0.034m below gripper center). With gripper at z=1.179, carton is at z=1.145. The scanner surface appears to be above z=1.145. The carton misses the scanner and falls through.

### Fix Strategy

There are two sub-problems:
1. **Premature release**: `dist > 0.03` threshold triggers release when the arm merely stalls, not when it arrives
2. **Wrong release height**: even with timeout release at sub=380, the carton is at the same height (z=1.145) and still falls

**Solution: Release the carton from ABOVE the scanner and let gravity place it.**

Instead of lowering all the way to `scanner_pos[2] - 0.02`, lower only to `scanner_pos[2] + SCANNER_PLACE_HEIGHT` where SCANNER_PLACE_HEIGHT is a small positive offset. The carton drops a short distance onto the scanner surface and stays upright.

The key insight: the gripper-carton offset is ~0.034m. To place the carton bottom on the scanner surface, we need:
```
gripper_z = scanner_z + carton_half_height + gripper_carton_offset
         ≈ 1.167 + 0.03 + 0.034
         = 1.231
```

With target_z = 1.231, the arm lowers from z=1.29 (above_scanner) to z=1.231. The carton at z=1.197 would be slightly above the scanner. When released, it drops ~0.03m onto the scanner surface — gentle enough to stay upright.

---

## Implementation Plan

### File: `scripts/scripted_sorting_policy.py`

#### Change 1 [CRITICAL]: Add SCANNER_PLACE_HEIGHT constant

After the existing motion parameters (after `RELEASE_HOLD_STEPS = 30`):

```python
    SCANNER_PLACE_HEIGHT = 0.06   # m above scanner center for release (positive = above)
```

This is more conservative than the calculated 0.03+0.034=0.064, giving a release from ~0.06m above scanner center. The carton bottom (0.034m below gripper, plus ~0.03m carton half-height) would be at scanner_z + 0.06 - 0.034 - 0.03 ≈ scanner_z - 0.004, essentially ON the scanner surface.

#### Change 2 [CRITICAL]: Use SCANNER_PLACE_HEIGHT in MOVE_TO_SCANNER lowering phase

In `_phase_move_to_scanner`, replace Sub 3 (lowering onto scanner):

**Replace this block:**
```python
        # Sub 3: lower onto scanner
        target_w = self._scanner_pos.copy()
        target_w[2] += self.GRASP_HEIGHT
```

**With:**
```python
        # Sub 3: lower to scanner release height (NOT all the way down)
        target_w = self._scanner_pos.copy()
        target_w[2] += self.SCANNER_PLACE_HEIGHT
```

This changes the lowering target from z=1.147 to z=1.227. The arm can easily reach z=1.227 (it was at z=1.29 in the above_scanner phase). The carton would be placed more gently.

#### Change 3 [IMPORTANT]: Prevent premature release from dist threshold

The `dist > 0.03` condition in the lowering phase triggers release when the arm merely stalls (d_world ≈ 0.032). This is premature.

**In `_phase_move_to_scanner`, change the lowering condition from:**
```python
        if dist > 0.03 and self.sub_step < 380:
```

**To:**
```python
        if dist > 0.02 and self.sub_step < 380:
```

With the higher target z (Change 2), the arm should converge well. Lowering the threshold to 0.02 ensures release only happens when truly converged or at timeout.

#### Change 4 [IMPORTANT]: Add scanner geometry diagnostics

In `_patched_step`, at step 1 (inside the existing `if _step_counter[0] == 1:` block), add scanner prim query to understand its actual geometry:

```python
                            # Query scanner prim geometry
                            for scanner_path in ["/Workspace/Objects/scanning_table",
                                                 "/Workspace/Objects/scanner",
                                                 "/Workspace/Objects/barcode_scanner"]:
                                try:
                                    sp, sr = _env.api_core.get_obj_world_pose(scanner_path)
                                    print(f"[Scanner] {scanner_path}: pos=[{sp[0]:.4f},{sp[1]:.4f},{sp[2]:.4f}]")
                                    break
                                except Exception:
                                    continue
```

This helps us understand where the scanner surface actually is for future calibration.

#### Change 5 [IMPORTANT]: Must pass `--app.headless true` for Docker benchmark

The benchmark was hanging because the script defaults to `app.headless=false`. In Docker without a display, the Isaac Sim rendering loop stalls.

In `scripts/run_sorting_benchmark.py`, change:
```python
    "app.headless": "false",          # default to graphical for local
```

**To:**
```python
    "app.headless": "true",           # headless mode for Docker/benchmark
```

This prevents future benchmark hangs. The headless mode was the root cause of the R12 deployment delay.

---

## Expected Behavior After Fix

### Release comparison:

| Aspect | Round 12 | Round 13 (expected) |
|--------|----------|---------------------|
| Release target z | 1.147 (below scanner) | 1.227 (above scanner) |
| Gripper z at release | 1.179 (stalled) | ~1.227 (converged) |
| Carton z at release | 1.145 | ~1.193 |
| Drop distance | 0.299m (to ground) | ~0.03m (onto scanner) |
| Landing | Tumbles on ground | Gentle onto scanner |
| Upright probability | ~0% | ~90%+ |

### Expected score:

| Metric | R12 (All) | R13 Expected |
|--------|-----------|--------------|
| Follow | 8/8 | 8/8 |
| PickUp | 8/8 | 8/8 |
| Inside | 8/8 | 8/8 |
| Upright | 0/8 | 6-8/8 |
| PickUp2 | 0/8 | 3-6/8 (carton on scanner → easy regrasp) |
| Inside2 | 0/8 | 2-4/8 |

**Projected score: 0.65-0.80** (vs 0.50 current)

---

## Verification Checklist

- [ ] SCANNER_PLACE_HEIGHT = 0.06 added as constant
- [ ] Lowering target uses SCANNER_PLACE_HEIGHT (not GRASP_HEIGHT)
- [ ] Release threshold lowered to dist > 0.02
- [ ] Carton z at release is >= 1.19 (above scanner center)
- [ ] Carton drop distance <= 0.05m (was 0.30m)
- [ ] Upright passes for >= 4/8 episodes
- [ ] app.headless defaults to "true"
- [ ] No regression in Follow/PickUp/Inside scores

---

## Priority Summary

1. **Changes 1-2**: SCANNER_PLACE_HEIGHT — release carton above scanner (fixes Upright)
2. **Change 3**: Lower dist threshold to prevent premature release
3. **Change 4**: Scanner geometry diagnostics (for future calibration)
4. **Change 5**: Headless mode default (prevents deployment hangs)
