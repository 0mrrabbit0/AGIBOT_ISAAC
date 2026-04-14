# Code Review & Modification Suggestions (Round 2)

**日期**: 2026-04-14
**审查者**: 本地 Claude Code (评估端)
**目标**: 云端 Claude Code 读取本文档后修改代码

---

## 上一轮修改评估

BUG 1 (INIT_BODY 顺序) 和 BUG 2 (速度参数) 已在 commit `a4b9dd0` 中修复，确认正确：
- INIT_BODY 已改为 FK 顺序 `[-1.045, 1.344, -0.319, 0.0, 1.57]` ✓
- 速度参数已调优 ✓
- FK 验证日志已添加 ✓
- bj1-bj4 hold 已验证生效: `[Patch] bj1-bj4 hold active: indices=[0, 1, 6, 11]` ✓
- bj5 初始值正确: `bj5: 1.5700` ✓

---

## 本轮运行结果：全部 0 分

```
Follow:          0.0
PickUpOnGripper: 0.0
Inside:          0.0
Upright:         0.0
PickUpOnGripper: 0.0
Inside:          0.0
E2E:             0
```

APPROACH 阶段 d 从 0.841 下降到 0.397 后停滞不前（500 步），然后 timeout 进入 GRASP，GRASP 阶段 d 从 0.309 继续停滞。机械臂完全无法接近目标。

---

## BUG 4 [致命]: ROBOT_BASE 缺少 z 高度 — 机械臂无法到达任何物体

**文件**: `scripts/scripted_sorting_policy.py` 第 42 行

**问题**: `ROBOT_BASE = [0.24469, 0.09325, 0.0]` 的 z=0.0，但 G2 机器人是站立的人形机器人，其 base_link 在世界坐标中应该在 ~0.83m 高度。

**运行时证据** (来自 FK 验证日志):
```
arm_base world (body_fk):  [0.565, 0.233, 0.230]   ← 肩膀在 z=0.23m，几乎在地面
EEF world (body_fk):       [0.934, 0.677, 0.066]   ← 夹爪在 z=0.07m，在地面上
carton world (实际):       [0.495, 0.881, 0.775]   ← 包裹在 z=0.78m (桌面高度)
scanner world (实际):      [0.929, 0.000, 1.167]   ← 扫码台在 z=1.17m
```

**根因**: body_fk 链从 ROBOT_BASE 开始，经过 bj1-bj5 关节到达 arm_base_link。但 ROBOT_BASE z=0.0 意味着链从地面开始。实际上 G2 机器人的 base_link（骨盆/腰部）在站立时距地面约 0.83m（腿部高度）。

**影响**:
- `world_to_arm_base()` 算出的目标在 arm 局部坐标中位置完全错误（z 差 ~0.83m）
- IK 求解的是一个不存在的目标位置 → 关节角度错误 → 机械臂往错误方向移动
- 这是机械臂无法接近包裹的**根本原因**

**修复方案**: 在 `run_sorting_benchmark.py` 的 `_create_env_pi` 中查询机器人实际世界位置，传递给 policy：

```python
# 在 create_env 的 post-creation 部分，scanner/bin 查询之后添加:
# Query robot base world position
robot_base_z = 0.0
try:
    # 尝试可能的机器人 prim 路径
    for robot_path in ["/Workspace/Robot", "/World/Robot", "/Workspace/robot"]:
        try:
            rpos, rrot = self.api_core.get_obj_world_pose(robot_path)
            robot_base_z = float(rpos[2])
            print(f"[Patch] Robot base world z={robot_base_z:.4f} (from {robot_path})")
            break
        except Exception:
            continue
    
    # 如果查不到机器人 prim，用 EEF 实际世界坐标反推
    if robot_base_z == 0.0:
        # 获取 env reset 后的第一帧 EEF 实际世界位置
        # 然后用 IKFKSolver 的 EEF local + body_fk 反算缺失的 z offset
        pass
except Exception as e:
    print(f"[Patch] WARNING: failed to query robot base: {e}")

if hasattr(self, 'policy') and robot_base_z > 0.1:
    self.policy.ROBOT_BASE[2] = robot_base_z
    print(f"[Patch] Updated policy ROBOT_BASE z={robot_base_z:.4f}")
```

**如果查不到 prim path，替代方案**：

根据仿真数据反推 z 偏移：
- 扫码台在 z=1.167，通常扫码台面在机器人胸前，合理
- 桌面在 z≈0.78，通常桌面在机器人腰部偏下，合理
- 如果肩膀（arm_base）应在 z≈1.06，当前 body_fk 输出 z=0.23，差值 0.83

可以在 policy 里硬编码一个合理估算：
```python
# 临时方案：如果 ROBOT_BASE z 仍为 0，用估算值
ROBOT_BASE = np.array([0.24469, 0.09325, 0.83])
```

但优先应通过仿真查询获取准确值。

---

## BUG 5 [严重]: 目标包裹选错 — policy 瞄准的不是评测器追踪的包裹

**运行时证据**:
```
评测器追踪: benchmark_carton_00364e8c → 位置 [0.302, 0.700, 0.779]
policy 瞄准: benchmark_carton_dd2ffc11 → 位置 [0.495, 0.881, 0.775]  (dict 第一个)
```

**问题**: 场景有 11 个 carton，policy 选了 dict 迭代的第一个，不是评测器要求的目标包裹。评测器通过 `problems.json` 指定了目标为 `benchmark_carton_00364e8c`。

**修复方案**: 从 `problems.json` 解析目标 carton 名称，传给 policy。

在 `run_sorting_benchmark.py` 的 `_create_env_pi` 中：

```python
# 在 carton_positions 之后，解析 problems.json 获取目标 carton 名称
target_carton_name = None
try:
    import json as _json
    import geniesim.utils.system_utils as _su
    problems_path = os.path.join(
        _su.benchmark_conf_path(), "llm_task",
        self.args.sub_task_name, str(instance_id), "problems.json",
    )
    if os.path.exists(problems_path):
        with open(problems_path) as f:
            problems = _json.load(f)
        # 在 problems 结构中搜索 Follow action 的 obj_name
        def _find_follow_target(obj):
            if isinstance(obj, dict):
                if obj.get("class_name") == "Follow":
                    params = obj.get("params", {})
                    names = params.get("obj_name_list", [])
                    if names:
                        return names[0]
                for v in obj.values():
                    r = _find_follow_target(v)
                    if r:
                        return r
            elif isinstance(obj, list):
                for item in obj:
                    r = _find_follow_target(item)
                    if r:
                        return r
            return None
        target_carton_name = _find_follow_target(problems)
        if target_carton_name:
            print(f"[Patch] Target carton from problems.json: {target_carton_name}")
except Exception as e:
    print(f"[Patch] WARNING: failed to parse problems.json: {e}")

# 用目标名称从 carton_positions 中选择
target_carton_pos = None
if target_carton_name and target_carton_name in carton_positions:
    target_carton_pos = carton_positions[target_carton_name]
    print(f"[Patch] Matched target carton: {target_carton_name} at {target_carton_pos}")
elif carton_positions:
    # fallback 到第一个
    name, pos = next(iter(carton_positions.items()))
    target_carton_pos = pos
    print(f"[Patch] WARNING: target not found, fallback to {name}")
```

然后将 `target_carton_pos` 传给 `self.policy.set_scene_positions()`（替换当前的 "028"/"020" 匹配逻辑）。

---

## BUG 6 [中等]: APPROACH 阶段不转腰 — 可能需要 bj5 旋转

**文件**: `scripts/scripted_sorting_policy.py` `_phase_approach()` 方法

**当前逻辑**: `bj5_hold = obs["bj5"]` — 保持当前腰部角度不动。

**问题**: 目标包裹可能不在当前 bj5 角度的手臂可达范围内。正确的 carton (`benchmark_carton_00364e8c` at [0.302, 0.700, 0.779]) 可能需要特定 bj5 角度才能让手臂够到。

**修复建议**: APPROACH 阶段应该在移动手臂的同时旋转 bj5 到 `_bj5_table`：

```python
def _phase_approach(self, obs: dict) -> np.ndarray:
    # Rotate bj5 toward table while approaching
    bj5_target = self._bj5_table
    new_bj5 = self._smooth_bj5(obs["bj5"], bj5_target, self.BJ5_SPEED)
    
    target_w = self._carton_pos.copy()
    target_w[2] += self.APPROACH_HEIGHT
    target_l = self.world_to_arm_base(target_w, obs["body"])
    new_joints, dist = self._move_right_toward(
        target_l, obs["r_eef_pos"], obs["r_eef_quat"],
        obs["arm_14"], step_size=self.EEF_STEP_FAST,
    )
    action = self._build_action(obs["left_arm"], new_joints, new_bj5)
    self._log(obs, target_w, "above_carton")
    
    if dist < 0.04 or self.sub_step > 500:
        self._set_phase("GRASP")
    return action
```

**注意**: 这个修改在 BUG 4 (z 偏移) 修好后才有意义。z 偏移是根本原因。

---

## 修改优先级

1. **BUG 4** (ROBOT_BASE z=0) — **根本原因**，必须立即修复
2. **BUG 5** (目标包裹选错) — 即使手臂能到达，瞄错了包裹也不得分
3. **BUG 6** (APPROACH 不转腰) — 优化可达性
4. 之前的 BUG 1-2 已修复 ✓

---

## 下次运行检查清单

- [ ] 查到 robot base world z > 0（应 ≈ 0.83m）
- [ ] arm_base world z ≈ 1.0-1.1m（肩膀高度合理）
- [ ] EEF world z ≈ 0.8-0.9m（初始时夹爪在腰部高度）
- [ ] 目标 carton 名称与 Follow 评测器一致（`benchmark_carton_00364e8c`）
- [ ] APPROACH 阶段 d 快速下降到 <0.04
- [ ] Follow 评分 > 0
