# Code Review & Modification Suggestions (Round 4)

**日期**: 2026-04-15
**审查者**: 本地 Claude Code (评估端)
**目标**: 云端 Claude Code 读取本文档后修改代码

---

## 上一轮修改评估

Commit `c3c894f`:
- problems.json 解析格式修复 ✓ — `{"Follow": "carton_id|bbox|gripper"}` 格式正确解析
- 路径 fallback ✓ — 成功从 hardcoded 路径加载 problems.json
- 目标 carton 正确选中 ✓ — `benchmark_carton_00364e8c` at [0.302, 0.700, 0.779]
- Robot link probe ✗ — **base_link z=0.0 覆盖了 0.83 fallback，导致 ROBOT_BASE z 又变回 0**

---

## 本轮运行结果：全部 0 分

```
Follow: 0.0 | PickUpOnGripper: 0.0 | Inside: 0.0 | Upright: 0.0 | PickUpOnGripper: 0.0 | Inside: 0.0
```

**好消息**: 目标 carton 修对后，APPROACH d 从 1.027 降到 0.223（比上轮 0.336 好很多）
**坏消息**: Probe 代码找到 base_link z=0.0 后覆盖了 0.83 fallback → ROBOT_BASE z=0 → body FK 世界坐标又错了

---

## BUG 10 [致命]: Probe 覆盖了正确的 z fallback

**文件**: `scripts/run_sorting_benchmark.py` — probe 代码块

**问题**: 代码在 robot_base_z=0.83 (fallback) 之后执行 probe，找到 `/Workspace/Robot/base_link` z=0.0，然后用 0.0 覆盖了 0.83。

**日志证据**:
```
[Patch] Robot base z fallback: 0.83          ← 正确
[Probe] /Workspace/Robot/base_link: z=0.0000 ← USD prim 本身在原点
[Probe] Using base_link z=0.0000             ← 覆盖了正确值！
[Patch] ROBOT_BASE updated: [0.24469 0.09325 0.     ]  ← z 又变回 0
```

**原因**: Isaac Sim 中 `/Workspace/Robot/base_link` 的 prim 位置是 0.0（USD 层级中的本地坐标），不代表仿真中的实际世界位置。所有 robot link prim 都报告 z=0.0。

**修复**: probe 代码应该只在 z > 0.1 时才覆盖 fallback。把 probe 块中的覆盖逻辑改为：

```python
for probe_path in [
    "/Workspace/Robot/base_link",
    "/Workspace/Robot/right_arm_link7",
    "/Workspace/Robot/idx67_arm_r_link7",
]:
    try:
        pos, _ = self.api_core.get_obj_world_pose(probe_path)
        print(f"[Probe] {probe_path}: z={float(pos[2]):.4f}")
        # Only use if z is meaningful (robot is standing, not at origin)
        if "base_link" in probe_path and float(pos[2]) > 0.1:
            robot_base_z = float(pos[2])
            print(f"[Probe] Using base_link z={robot_base_z:.4f}")
    except Exception:
        continue
```

并且把 `self.policy.ROBOT_BASE[2] = robot_base_z` 移到 probe 循环之后、与其他 policy 设置一起做（而不是在 probe 循环内部做，那会绕过后面的 `ROBOT_BASE updated` 逻辑）。

**最简修复**: 直接删除整个 probe 代码块。0.83 作为 fallback 已经接近正确。probe 在当前 Isaac Sim 环境下无法获取有用信息。删掉后代码更简洁可靠：

```python
# 删除 for probe_path in [...] 整个块
# 保留 robot_base_z = 0.83 fallback
```

---

## 验证: z=0.83 让手臂能到达目标

上轮运行 (ROBOT_BASE z=0.83):
- arm_base z=1.06, EEF z=0.90
- 目标 carton z=0.78 + 0.15 = 0.93
- 这些高度都在合理范围，手臂应该能到达

本轮运行 (ROBOT_BASE z=0.0，因 probe bug):
- arm_base z=0.23, EEF z=0.07
- 目标 carton z=0.78 + 0.15 = 0.93
- z 差距太大 → IK 无法收敛，d 停在 0.223

**预期**: 修复后（z=0.83 + 正确目标 carton），arm 应该能到达 d<0.04

---

## 修改优先级

1. **BUG 10** (probe 覆盖 z) — **唯一需要修复的问题**，删掉 probe 块或加 > 0.1 检查

---

## 下次运行检查清单

- [ ] `ROBOT_BASE updated: [0.244, 0.093, 0.83]` （z=0.83，不是 0.0）
- [ ] `arm_base (INIT): z ≈ 1.06` （肩膀高度）
- [ ] `EEF world: z ≈ 0.90` （初始夹爪高度）
- [ ] APPROACH d 快速下降到 <0.04
- [ ] Follow score > 0
- [ ] 进入 GRASP 阶段后关闭夹爪
