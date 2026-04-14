# Code Review & Modification Suggestions (Round 3)

**日期**: 2026-04-14
**审查者**: 本地 Claude Code (评估端)
**目标**: 云端 Claude Code 读取本文档后修改代码

---

## 上一轮修改评估

Commit `56821ae` 修复了 3 个 BUG：
- Robot base z 查询 + 0.83 fallback ✓ — arm_base 现在在 z=1.06 (肩高)，EEF 在 z=0.90 (腰高)，合理
- problems.json 解析 ✗ — **解析格式错误**，未找到目标，仍 fallback 到错误 carton
- APPROACH 阶段 bj5 旋转 ✓ — 代码正确

---

## 本轮运行结果：全部 0 分（与上轮相同）

```
Follow: 0.0 | PickUpOnGripper: 0.0 | Inside: 0.0 | Upright: 0.0 | PickUpOnGripper: 0.0 | Inside: 0.0
```

APPROACH 阶段 d 从 0.576 下降到 0.336 后停滞 500 步。机械臂仍无法接近目标。

**根因分析**: 两个问题叠加：
1. 瞄错包裹（第一个 carton 而非评测目标）
2. 错误的包裹距 arm_base 0.67m，刚好在 7-DOF 手臂极限边缘，无法有效逼近

---

## BUG 7 [致命]: problems.json 解析格式不匹配

**文件**: `scripts/run_sorting_benchmark.py` — `_find_follow_target()` 函数

**问题**: 解析代码期望格式 `{"class_name": "Follow", "params": {"obj_name_list": [...]}}` ，但 problems.json 实际格式是：

```json
{
  "Follow": "benchmark_carton_00364e8c|[0.2,0.2,0.2]|right_gripper"
}
```

**实际 problems.json 路径**: `/workspace/genie_sim/source/geniesim/benchmark/config/llm_task/sorting_packages/0/problems.json`

**实际结构** (嵌套在 `problem1.Acts[0].ActionList[0].ActionSetWaitAny[0]` 下):
```json
{"Follow": "benchmark_carton_00364e8c|[0.2,0.2,0.2]|right_gripper"}
```

目标 carton ID 是 `|` 分隔字符串的第一个字段。

**修复**: 替换 `_find_follow_target` 函数：

```python
def _find_follow_target(obj):
    """Recursively find Follow target in problems.json.
    
    Format: {"Follow": "carton_id|bbox|gripper_id"}
    """
    if isinstance(obj, dict):
        if "Follow" in obj:
            val = obj["Follow"]
            if isinstance(val, str):
                return val.split("|")[0]  # first field is carton name
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
```

---

## BUG 8 [中等]: benchmark_conf_path() 可能被 stub 拦截

**文件**: `scripts/run_sorting_benchmark.py`

**问题**: 代码用 `import geniesim.utils.system_utils as _su` 然后调用 `_su.benchmark_conf_path()`。但在 FallbackImporter 环境下，如果这个模块的某些依赖被 stub 了，函数可能返回空或异常。

**验证**: 日志中没有 `[Patch] Target carton from problems.json:` 的输出，说明 problems.json 确实没被正确解析。可能是路径错误或解析失败。

**修复**: 添加 hardcoded 路径作为 fallback：

```python
# 在 problems.json 查找逻辑中添加:
problems_paths = []
try:
    import geniesim.utils.system_utils as _su
    problems_paths.append(os.path.join(
        _su.benchmark_conf_path(), "llm_task",
        self.args.sub_task_name, str(instance_id),
        "problems.json",
    ))
except Exception:
    pass
# Hardcoded fallback path
problems_paths.append(os.path.join(
    genie_sim_root, "source", "geniesim", "benchmark", "config",
    "llm_task", self.args.sub_task_name, str(instance_id),
    "problems.json",
))

# 在 problems_paths 中找到第一个存在的
problems_data = None
for pp in problems_paths:
    if os.path.exists(pp):
        with open(pp) as f:
            problems_data = _json.load(f)
        print(f"[Patch] Loaded problems.json from {pp}")
        break

if problems_data:
    target_carton_name = _find_follow_target(problems_data)
```

注意: `genie_sim_root` 变量在文件顶部已定义为 `"/workspace/genie_sim"`。

---

## BUG 9 [低]: robot base z 偏移可能不精确

**文件**: `scripts/run_sorting_benchmark.py` + `scripts/scripted_sorting_policy.py`

**现状**: `/Workspace/Robot` prim 存在但 z=0.0（机器人 USD root 在地面），fallback 用 0.83。

**验证**: 目前无法直接验证 0.83 是否正确。建议添加诊断：在 `_create_env_pi` 中查询 EEF link 的实际世界位置，与 body_fk 输出对比：

```python
# 在 create_env 中，robot_base_z 之后添加探针:
for probe_path in [
    "/Workspace/Robot/right_arm_link7",
    "/Workspace/Robot/idx67_arm_r_link7",
    "/Workspace/Robot/base_link",
]:
    try:
        pos, _ = self.api_core.get_obj_world_pose(probe_path)
        print(f"[Probe] {probe_path}: world z={float(pos[2]):.4f}")
    except Exception:
        continue
```

如果能查到 `base_link` 的 z 值，直接用作 ROBOT_BASE z 代替 0.83 估算值。

---

## 验证: 正确目标距离更近

当前错误目标 `benchmark_carton_dd2ffc11` [0.495, 0.881]:
- 距 arm_base [0.565, 0.233, 1.060] 水平距离 = 0.66m ← 在臂长极限(~0.65m)
- IK 无法收敛 → d 停在 0.336

正确目标 `benchmark_carton_00364e8c` [0.302, 0.700]:
- 距 arm_base 水平距离 ≈ 0.54m ← 在可达范围内
- 修复 BUG 7 后应该能到达

---

## 修改优先级

1. **BUG 7** (problems.json 解析格式) — **根本原因**，修复后选对目标包裹
2. **BUG 8** (路径 fallback) — 确保能找到 problems.json 文件
3. **BUG 9** (robot z 验证) — 添加探针日志，下次运行验证

---

## 下次运行检查清单

- [ ] `[Patch] Target carton from problems.json: benchmark_carton_00364e8c` 出现
- [ ] `[Patch] Matched target: benchmark_carton_00364e8c at [0.302, 0.700, ...]` 出现
- [ ] carton bj5 角度对应正确包裹位置
- [ ] APPROACH d 快速下降到 <0.04（200 步内）
- [ ] Follow score > 0
- [ ] 如有 probe 日志，检查 base_link z 值
