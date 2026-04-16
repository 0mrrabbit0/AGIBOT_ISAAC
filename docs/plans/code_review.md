# Code Review & Modification Suggestions (Round 12)

**Date**: 2026-04-16
**Reviewer**: Local Claude Code (Evaluation)
**Target**: Cloud Claude Code reads this doc and modifies code

---

## Round 11 Result: Timing fix worked, but Upright regressed. Average = 0.3333

### Scores (8 episodes)

| Episode | Carton | Follow | PickUp | Inside | Upright | PickUp2 | Inside2 |
|---------|--------|--------|--------|--------|---------|---------|---------|
| 1       | Yellow | **1** | **1** | **1** | 0 | 0 | 0 |
| 2       | Yellow | **1** | **1** | **1** | 0 | 0 | 0 |
| 3       | Yellow | **1** | **1** | **1** | 0 | 0 | 0 |
| 4       | Yellow | **1** | **1** | **1** | 0 | 0 | 0 |
| 5       | Black  | **1** | 0 | 0 | 0 | 0 | 0 |
| 6       | Black  | **1** | 0 | 0 | 0 | 0 | 0 |
| 7       | Black  | **1** | 0 | 0 | 0 | 0 | 0 |
| 8       | Black  | **1** | 0 | 0 | 0 | 0 | 0 |

**Average: 0.3333** (16/48 total points)

### What Round 11 Changed

1. **MOVE_TO_SCANNER timing cut** — completes in ~411 sub-steps (was 731). Step 780 vs 1100. This worked mechanically.
2. **REGRASP targets real carton** — correctly reads `_shared_state["real_carton_world"]` instead of scanner position.
3. **Faster waist rotation** — BJ5_SPEED=0.06 (was 0.04).

### Why Upright Went from 1/4 to 0/4

The timing fix saved steps but did NOT fix the fundamental problem: **the arm cannot reach the scanner surface**. The carton is released at x=0.874, but the scanner center is at x=0.929 (deficit = 0.055m). The carton falls 0.27m to the ground regardless of timing. Whether it lands upright is pure physics luck (~25% chance).

R10 got lucky on 1 episode. R11 got unlucky on all 4. The variance is expected because we're dropping the carton from 0.27m height.

### Why REGRASP Still Fails

Carton lands at x=0.891 (R11) or x=0.801 (R10) — variable depending on release dynamics. The arm's horizontal reach is ~0.869m from robot base. When the carton lands near x=0.89, it's beyond reach (d_world=0.089, arm stalls).

---

## ROOT CAUSE: Arm reach is 0.06m too short for scanner

The arm_base_link is at a fixed position (determined by bj1-bj4 held constant). The arm can extend ~0.56m from arm_base. This is insufficient to reach the scanner at x=0.929.

**Solution: Body lean via bj2 adjustment.** bj2 is a pitch joint that tilts the torso forward. Increasing bj2 by 0.27 rad moves the arm_base_link forward by approximately 0.06m in the current facing direction. This closes the gap to the scanner.

The body lean is controlled via `_patched_step`'s `set_joint_positions`. Currently bj1-bj4 are ALL held at fixed initial values. We modify this to allow bj2 to change dynamically, controlled by the policy via `_shared_state`.

---

## Implementation Plan

### File: `scripts/run_sorting_benchmark.py`

#### Change 1 [CRITICAL]: Add `desired_bj2` to shared state initialization

After line 533 (`_shared_state = {"real_gripper_world": None}`), add:

```python
_shared_state = {"real_gripper_world": None, "desired_bj2": _bs[3]}
```

Here `_bs[3]` is the initial bj2 value (1.344 from body_state `[bj5,bj4,bj3,bj2,bj1]`, so `_bs[3]` = bj2).

#### Change 2 [CRITICAL]: Add dynamic bj2 ramping in `_patched_step`

Inside `_patched_step`, BEFORE the `set_joint_positions` call (before line 559), add bj2 ramping logic:

```python
                        # Dynamic bj2 control — ramp toward desired value
                        _desired_bj2 = _shared_state.get("desired_bj2", _body_hold[1])
                        if abs(_desired_bj2 - _body_hold[1]) > 0.005:
                            _bj2_delta = min(0.01, abs(_desired_bj2 - _body_hold[1]))
                            _body_hold[1] += _bj2_delta if _desired_bj2 > _body_hold[1] else -_bj2_delta
                            if _step_counter[0] % 30 == 0:
                                print(f"[Patch] bj2 ramping: current={_body_hold[1]:.3f} target={_desired_bj2:.3f}")
```

This goes AFTER the `_body_indices_cache` resolution block (line 550-558) and BEFORE the `set_joint_positions` call (line 559-564). The ramping rate is 0.01 rad/step, so 0.27 rad lean takes ~27 steps (about 1 second at 30Hz sim).

**`_body_hold` is a mutable list** defined at line 528. Modifying `_body_hold[1]` inside the closure works because Python closures capture references to mutable objects.

#### Change 3 [CRITICAL]: Query arm_base prim EVERY step (not just step 1)

Currently arm_base prim pos/rot is only queried at step 1 (inside `if _step_counter[0] == 1:` block, lines 591-647). When bj2 changes, the arm_base position/rotation changes. The policy's `_world_target_to_solver_frame` uses `_shared_state["arm_base_rot"]` to compute local error direction. If this is stale, IK targets will have wrong direction.

**Add after the carton query block** (after line 589), OUTSIDE the `if _step_counter[0] == 1:` block:

```python
                        # Query arm_base prim EVERY step (position changes with body lean)
                        if "arm_base_prim" in _shared_state:
                            try:
                                _ab_p, _ab_r = _env.api_core.get_obj_world_pose(
                                    _shared_state["arm_base_prim"]
                                )
                                _shared_state["arm_base_pos"] = [
                                    float(_ab_p[i]) for i in range(3)
                                ]
                                _shared_state["arm_base_rot"] = [
                                    float(_ab_r[i]) for i in range(4)
                                ]
                            except Exception:
                                pass
```

This uses the arm_base prim path discovered at step 1 (`_shared_state["arm_base_prim"]`). The step 1 code that finds the prim path stays unchanged. We just add continuous tracking after step 1.

---

### File: `scripts/scripted_sorting_policy.py`

#### Change 4 [CRITICAL]: Add body lean constants

Add after the existing motion parameters section (after line 81, `RELEASE_HOLD_STEPS = 30`):

```python
    # ── Body lean for extended reach ─────────────────────────────────
    BJ2_INIT = 1.344          # initial bj2 value (from body_state)
    BJ2_LEAN_OFFSET = 0.27    # radians forward lean (~0.06m reach extension)
    BJ2_LEAN = BJ2_INIT + BJ2_LEAN_OFFSET  # = 1.614
```

#### Change 5 [CRITICAL]: Set desired_bj2 in `_set_phase`

Replace the `_set_phase` method (line 432-436) with:

```python
    def _set_phase(self, name: str) -> None:
        print(f"[Policy] {self.phase} → {name} "
              f"(step={self.step_count}, sub={self.sub_step})")
        self.phase = name
        self.sub_step = 0
        # Body lean control via shared state
        if self._shared_state is not None:
            if name in ("APPROACH", "GRASP", "MOVE_TO_SCANNER", "REGRASP"):
                self._shared_state["desired_bj2"] = self.BJ2_LEAN
            elif name in ("MOVE_TO_BIN", "RETURN", "DONE"):
                self._shared_state["desired_bj2"] = self.BJ2_INIT
```

**Lean schedule:**
- **APPROACH**: Lean forward. Extends reach for all cartons. Yellow cartons don't need it (IK closed-loop handles overshoot), but black cartons NEED the extra 0.06m reach.
- **GRASP**: Keep lean. Carton is being grasped and lifted.
- **MOVE_TO_SCANNER**: Keep lean. This is the critical phase — lean provides the extra 0.06m to reach the scanner surface.
- **REGRASP**: Keep lean. Carton is near scanner, needs extended reach.
- **MOVE_TO_BIN**: Lean back to neutral. The bin is in a different direction; lean isn't needed and could interfere with rotation stability.
- **RETURN, DONE**: Neutral position.

**Why lean during APPROACH (not just MOVE_TO_SCANNER):**
The black carton stalls at d_world=0.076m from target in APPROACH. The arm reach deficit is ~0.069m. Body lean adds ~0.06m reach in the facing direction. This brings the deficit to ~0.01m, which should be within the gripper's grasp range (gripper close threshold is `real_dist < 0.035`).

The lean ramps at 0.01 rad/step, so it takes ~27 steps (under 1 second). APPROACH starts at step 30 and lasts 85-300 steps. The lean is fully established well before the arm reaches the carton.

For yellow cartons, the lean is harmless because:
1. The closed-loop relative-error IK adjusts targets every step
2. The arm_base prim is queried every step (Change 3), so the IK always uses the correct arm_base position
3. Yellow carton APPROACH completes quickly (dist < 0.03 within ~85 steps)

---

## Expected Behavior After Fix

### Timeline (yellow carton with body lean):

```
Step   0-30:   INIT (30 steps, bj2=1.344)
Step  30-57:   APPROACH starts, bj2 ramps 1.344 → 1.614 (27 steps)
Step  57-115:  APPROACH continues at full lean, arm reaches carton easily
Step 115-369:  GRASP (same as before, lean maintained)
Step 369-780:  MOVE_TO_SCANNER (with lean!)
               Arm_base is ~0.06m farther forward
               Arm can now reach x ≈ 0.929 (scanner center!)
               Carton placed ON scanner surface, not dropped from 0.27m height
               Upright: carton is ON the surface → trivially passes (<5° angle)
Step 780-1100: REGRASP (with lean)
               Carton is ON scanner → easy to re-grasp in place
               PickUp2 should trigger
Step 1100+:    MOVE_TO_BIN (lean back), rotate, extend, drop
               Inside2 possible
```

### Key difference from Round 11:

| Aspect | Round 11 | Round 12 (expected) |
|--------|----------|---------------------|
| Scanner reach | x=0.874 (0.055m short) | x≈0.929 (ON scanner) |
| Carton drop height | 0.27m to ground | ~0m (placed on surface) |
| Upright probability | ~25% (random fall) | ~90%+ (placed flat) |
| REGRASP target distance | 0.089m (beyond reach) | ~0m (carton on scanner, in reach) |
| Black carton APPROACH | stalls at 0.076m | closes to ~0.01m (within grasp) |

### Expected score:

| Metric | R11 (Yellow) | R12 Expected (Yellow) | R11 (Black) | R12 Expected (Black) |
|--------|-------------|----------------------|-------------|---------------------|
| Follow | 4/4 | 4/4 | 4/4 | 4/4 |
| PickUp | 4/4 | 4/4 | 0/4 | 2-4/4 (lean helps reach) |
| Inside | 4/4 | 4/4 | 0/4 | 0-2/4 |
| Upright | 0/4 | 3-4/4 | 0/4 | 0/4 |
| PickUp2 | 0/4 | 2-4/4 | 0/4 | 0/4 |
| Inside2 | 0/4 | 1-3/4 | 0/4 | 0/4 |

**Projected score: 0.42-0.58** (vs 0.3333 current)

---

## Verification Checklist

- [ ] `_shared_state["desired_bj2"]` initialized to 1.344 in `run_sorting_benchmark.py`
- [ ] bj2 ramps smoothly in `_patched_step` (check log: `[Patch] bj2 ramping`)
- [ ] Arm_base prim queried every step (not just step 1)
- [ ] Arm reaches scanner center (x ≈ 0.929, was x=0.874)
- [ ] Carton placed on scanner surface (not dropped to ground)
- [ ] Upright passes for >= 3/4 yellow episodes (was 0/4)
- [ ] REGRASP succeeds (carton on scanner, easy to re-grasp)
- [ ] PickUp2 triggers for >= 1 yellow episode (was 0/4)
- [ ] Black carton PickUp improved (arm reaches closer, was stalled at 0.076m)
- [ ] No regression in Follow or Inside scores
- [ ] Robot remains stable during lean (no tipping or oscillation)

---

## Priority Summary

1. **Changes 1-3** (run_sorting_benchmark.py): Enable dynamic bj2 control infrastructure
2. **Changes 4-5** (scripted_sorting_policy.py): Control body lean per phase

All 5 changes are CRITICAL and must be implemented together. The body lean without continuous arm_base tracking (Change 3) will cause IK directional errors. The policy lean control (Changes 4-5) without the _patched_step infrastructure (Changes 1-2) does nothing.

---

## Not Addressed (Round 13+)

- **Fine-tuning BJ2_LEAN_OFFSET**: 0.27 rad is an estimate. If the real robot geometry differs from our FK model, we may need to adjust.
- **Gripper RPY correction**: After lean, the gripper approaches at a ~15 degree tilt. This may or may not affect carton placement quality.
- **MOVE_TO_BIN accuracy**: Depends on PickUp2 working first. Bin placement not yet verified.
