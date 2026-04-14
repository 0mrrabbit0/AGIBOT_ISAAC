# Code Review & Modification Suggestions

**日期**: 2026-04-14
**审查者**: 本地 Claude Code (评估端)
**目标**: 云端 Claude Code 读取本文档后修改代码

---

## 评估状态

上一次运行（有图形界面）成功启动 Isaac Sim，ScriptedSortingPolicy 被加载，carton prims 被找到，scene positions 被传递给 policy。
机械臂在 APPROACH 阶段开始移向目标包裹，但 240 步内距离只从 0.924 缩小到 0.514（未达到 <0.04 的完成阈值）。bj1-bj4 hold 因 `robot_joint_indices` 未初始化而未生效（已有 lazy init 修复，需验证）。

---

## BUG 1 [严重]: INIT_BODY 关节顺序错误

**文件**: `scripts/scripted_sorting_policy.py` 第 44 行

**问题**: `INIT_BODY` 定义为 G2_WAIST 顺序 `[bj5, bj4, bj3, bj2, bj1]`，但 `body_fk()` 期望 URDF/FK 顺序 `[bj1, bj2, bj3, bj4, bj5]`。

**当前值**:
```python
INIT_BODY = np.array([1.57, 0.0, -0.31939525311, 1.34390352404, -1.04545222194])
# 实际含义: [bj5=1.57, bj4=0.0, bj3=-0.319, bj2=1.344, bj1=-1.045]
```

**影响范围**:
1. `body_fk(INIT_BODY)` — bj1 用了 1.57 (实为 bj5 值)，FK 输出完全错误
2. `_compute_bj5_for_target()` 第 200-217 行 — 用 INIT_BODY.copy() 做 bj5 sweep，bj1-bj4 值是错的，计算出的 bj5 目标角度不正确
3. `INIT_BODY[4]` 被当作 bj5 初始值用于 `_phase_return` (第 739 行) 和 `_phase_done` (第 751 行)，但 `INIT_BODY[4] = -1.045` 实为 bj1 值，bj5 实际初始值是 1.57
4. `_parse_obs()` 第 345 行的 fallback `self.INIT_BODY[4]` 也是错的

**修复**: 将 INIT_BODY 改为 FK 顺序 `[bj1, bj2, bj3, bj4, bj5]`:

```python
# Body joints in FK/URDF order: [bj1, bj2, bj3, bj4, bj5]
# Derived from G2_STATES_4 body_state [bj5,bj4,bj3,bj2,bj1] = [1.57, 0.0, -0.319, 1.344, -1.045]
INIT_BODY = np.array([-1.04545222194, 1.34390352404, -0.31939525311, 0.0, 1.57])
```

修复后 `INIT_BODY[4] = 1.57 = bj5 初始值` ✓，`body_fk(INIT_BODY)` 输入正确 ✓

**注意**: `_get_body_joints()` 在运行时用 `states[16:21]` (已经是 FK 顺序 [bj1..bj5]) 覆盖所有值，所以运行时 FK 不受影响。但离线计算（bj5 sweep、init 日志）全部错误。

---

## BUG 2 [中等]: 机械臂逼近速度过慢

**文件**: `scripts/scripted_sorting_policy.py`

**现象**: APPROACH 阶段 240 步只移动了 0.41m (0.924→0.514)，有效速度约 0.0017 m/step，远低于 `EEF_STEP_FAST = 0.012 m/step`。

**根因分析**: 
- `MAX_JOINT_DELTA = 0.10` (第 57 行) — 每步关节增量上限太紧，IK 解算出的大角度变化被截断
- `JOINT_SMOOTH_ALPHA = 0.5` (第 56 行) — 平滑系数进一步减半实际移动量
- 综合效果: 实际每步只能执行 `0.10 * 0.5 = 0.05 rad` 的关节变化，导致末端移动缓慢
- BUG 1 导致 bj5 计算错误，可能使目标在 arm_base_link 中位置偏离最优方向，IK 收敛更差

**修复建议** (在 `scripted_sorting_policy.py` 常量区):
```python
JOINT_SMOOTH_ALPHA = 0.7    # 原 0.5 → 0.7 (更快跟随)
MAX_JOINT_DELTA = 0.15      # 原 0.10 → 0.15 (允许更大关节变化)
EEF_STEP_FAST = 0.018       # 原 0.012 → 0.018 (更大笛卡尔步长)
IK_POS_TOLERANCE = 0.12     # 原 0.08 → 0.12 (接受稍差的 IK 解)
```

**优先级**: 先修 BUG 1 再调参。BUG 1 修好后 bj5 计算正确，目标位置在 arm 可达范围内最优位置，IK 收敛会自然改善。

---

## BUG 3 [低]: 目标包裹匹配逻辑失效

**文件**: `scripts/run_sorting_benchmark.py` 第 401-409 行 + `scripts/scripted_sorting_policy.py` 第 417-435 行

**问题**: 运行时 carton prim 名称是哈希 ID（如 `benchmark_carton_dd2ffc11`），代码用 `"028" in name` 或 `"020" in name` 匹配永远失败，总是 fallback 到第一个 carton。

**影响**: 当场景有多个 carton 时可能抓错目标。当前 instance 0 只有一个 carton 所以不影响。

**修复建议**: 
1. 在 `run_sorting_benchmark.py` 的 `_create_env_pi` 中，解析 `scene_info.json` 的 layout 来获取 carton 的 USD 类型名：

```python
# 在 _create_env_pi 中，carton_positions 之后添加:
# 解析 scene_info.json 获取 carton USD 类型映射
carton_type_map = {}  # {prim_name: usd_type}  例如 {"benchmark_carton_dd2ffc11": "carton_028"}
import json as _json
import geniesim.utils.system_utils as _su
scene_info_path = os.path.join(
    _su.benchmark_conf_path(), "llm_task",
    self.args.sub_task_name, str(instance_id), "scene_info.json",
)
if os.path.exists(scene_info_path):
    with open(scene_info_path) as f:
        si = _json.load(f)
    for obj_id, obj_data in si.get("layout", {}).items():
        if "carton" in obj_id:
            carton_type_map[obj_id] = obj_data.get("usd", "")
    if hasattr(self, 'policy') and hasattr(self.policy, 'set_carton_type_map'):
        self.policy.set_carton_type_map(carton_type_map)
```

2. 在 `scripted_sorting_policy.py` 中基于 USD 类型匹配目标

**暂时可忽略** — 当前 fallback 到第一个 carton 能工作。

---

## 建议 1: 验证 bj1-bj4 hold

**文件**: `scripts/run_sorting_benchmark.py` 第 429-461 行

lazy init 的 `_body_indices_cache` 逻辑已写好，但未经运行验证。下次运行时注意看日志：
- 应该出现 `[Patch] bj1-bj4 hold active: indices=[...]`
- 如果没出现，说明 `robot_joint_indices` 属性仍不可用，需要检查是否在 `reset()` 之后才有

---

## 建议 2: 添加运行时 body FK 验证日志

**文件**: `scripts/scripted_sorting_policy.py` 的 `_log_init()` 方法 (第 385-405 行)

建议在 `_log_init` 中添加 FK 验证，对比 FK 输出与 IKFKSolver 输出，确认坐标系一致：

```python
# 在 _log_init 的 print 块末尾添加:
if self.ikfk_solver is not None:
    fk_eef = self.ikfk_solver.compute_eef(obs["arm_14"])
    print(f"  IKFKSolver EEF(R): {fk_eef['right'][:3]}")
    print(f"  FK body chain EEF world: {eef_w}")
    print(f"  FK body chain test (init): {self.body_fk(self.INIT_BODY)[:3,3] + self.ROBOT_BASE}")
```

---

## 修改优先级

1. **BUG 1** (INIT_BODY 顺序) — **必须立即修复**，影响所有 bj5 计算和 FK 输出
2. **BUG 2** (速度参数) — 修完 BUG 1 后调参
3. 建议 2 (FK 验证日志) — 有助于确认修复效果
4. BUG 3 (carton 匹配) — 暂不影响，可后续处理

---

## 下次运行检查清单

- [ ] `[Patch] bj1-bj4 hold active` 日志是否出现
- [ ] `[Init] Calibration` 中 body_joints 打印值是否合理（bj5 应 ≈ 1.57）
- [ ] APPROACH 阶段 d 下降速度是否加快（目标 <0.04 在 200 步内）
- [ ] 是否进入 GRASP 阶段
- [ ] bj5 sweep 计算的 carton/scanner/bin 角度是否合理
