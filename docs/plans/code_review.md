# Code Review & Modification Suggestions (Round 5)

**日期**: 2026-04-15
**审查者**: 本地 Claude Code (评估端)
**目标**: 云端 Claude Code 读取本文档后修改代码

---

## 上一轮修改评估

Commit `9c2af8d`: 删除了 probe 块。但运行时 probe 仍然覆盖了 z（因为 probe 代码在 robot_base_z fallback 之后执行，找到 base_link z=0 后覆盖了 0.83）。

**BUG 10 已修复** ✓ — probe 块已删除

---

## 本轮运行结果：全部 0 分

但有巨大进步：
```
APPROACH: d=0.275 → d<0.04 in 52 steps! (之前要 500 步且不收敛)
GRASP: 成功进入，夹爪关闭 ✓
MOVE_TO_SCANNER: 转腰+移动到扫码台 ✓
```

然而 Follow 评分 = 0。StepOut 在 600 步超时后触发，Follow 被取消。

---

## 关键发现：Follow 评测器使用 `/genie/` prim 路径

**来源**: 容器内 `/workspace/genie_sim/source/geniesim/plugins/ader/action/custom/follow.py`

Follow 评测器获取夹爪真实世界位置的方式:
```python
link_prim_path = "/genie/gripper_r_center_link"  # 不是 /Workspace/Robot/
g_pos, _ = self.get_world_pose(link_prim_path)  # = api_core.get_obj_world_pose()
```

Follow 判定逻辑:
```python
# bbox = [0.2, 0.2, 0.2] → 以包裹中心为原点 ±0.1m 的 AABB
aa, bb = self.get_obj_aabb(obj_name, self.bbox)  # 取包裹世界位姿旋转后的 AABB
return self.aabb_contains_point(gripper_pos, (aa, bb))  # 夹爪在 AABB 内？
```

**所以 Follow 失败意味着: 仿真中 `/genie/gripper_r_center_link` 的真实世界位置不在包裹 ±0.1m 范围内**

---

## BUG 11 [致命]: z=0.83 偏移量不正确，导致真实夹爪位置偏离目标

**文件**: `scripts/run_sorting_benchmark.py` (z=0.83 fallback) + `scripts/scripted_sorting_policy.py` (body FK)

**问题**: 我们的 body FK 用 ROBOT_BASE z=0.83 计算世界坐标。如果这个值不准确，`world_to_arm_base()` 给 IK 的目标位置就是错的，IK 解算出的关节角度使真实夹爪到达错误位置。

**诊断方案**: 在 `_patched_step` 或 `_create_env_pi` 中添加代码，查询夹爪的**真实世界位置**并与 FK 输出对比：

在 `run_sorting_benchmark.py` 的 `_patched_step` 中添加诊断（每 30 步打印一次）:

```python
_step_counter = [0]

def _patched_step(action):
    result = _orig_step(action)
    _step_counter[0] += 1
    
    # Hold bj1-bj4 (existing code)
    if not _body_indices_cache:
        if hasattr(_env, 'robot_joint_indices'):
            _body_indices_cache.extend(
                _env.robot_joint_indices[v] for v in _body_names
            )
            print(f"[Patch] bj1-bj4 hold active: indices={_body_indices_cache}")
    if _body_indices_cache:
        _env.api_core.set_joint_positions(
            [float(v) for v in _body_hold],
            joint_indices=_body_indices_cache,
            is_trajectory=True,
        )
    
    # Diagnostic: query real gripper world position
    if _step_counter[0] % 30 == 1:
        try:
            real_pos, _ = _env.api_core.get_obj_world_pose(
                "/genie/gripper_r_center_link"
            )
            print(f"[Diag] step={_step_counter[0]} "
                  f"real_gripper=[{real_pos[0]:.4f},{real_pos[1]:.4f},{real_pos[2]:.4f}]")
        except Exception as e:
            print(f"[Diag] gripper query failed: {e}")
    
    return result
```

**这是唯一确定 z 偏移是否正确的方法。** 对比 `real_gripper` 和 policy 打印的 `eef_w` 值：
- 如果 z 差距 ≈ 0 → z=0.83 正确，问题在别处
- 如果 z 差距 ≈ X → ROBOT_BASE z 应调整为 0.83 - X
- 如果完全查不到 `/genie/gripper_r_center_link` → 需要换其他路径

**额外**: 可以同时打印包裹与夹爪的距离来验证 Follow 判定:

```python
    # Also print distance to target carton
    if _step_counter[0] % 30 == 1:
        try:
            real_pos, _ = _env.api_core.get_obj_world_pose(
                "/genie/gripper_r_center_link"
            )
            # target carton name stored earlier
            if target_carton_name:
                carton_prim = None
                for parent in ["/Workspace/Objects"]:
                    cp = f"{parent}/{target_carton_name}"
                    try:
                        c_pos, _ = _env.api_core.get_obj_world_pose(cp)
                        carton_prim = cp
                        break
                    except:
                        continue
                if carton_prim:
                    dist = np.sqrt(sum((float(real_pos[i])-float(c_pos[i]))**2 
                                       for i in range(3)))
                    print(f"[Diag] real_grip-carton dist={dist:.4f} "
                          f"carton=[{c_pos[0]:.3f},{c_pos[1]:.3f},{c_pos[2]:.3f}]")
        except Exception as e:
            print(f"[Diag] query failed: {e}")
```

---

## 修改优先级

1. **诊断代码** (BUG 11) — 添加到 `_patched_step`，查询 `/genie/gripper_r_center_link` 真实世界位置
2. 不要改其他东西 — 先确定 z 偏移问题的准确大小

---

## 重要：机器人 prim 根路径是 `/genie/`

之前我们以为是 `/Workspace/Robot/`，但 Follow 评测器用的是 `/genie/`。所有机器人 link 查询应使用 `/genie/` 前缀：
- `/genie/base_link`
- `/genie/gripper_r_center_link`
- `/genie/gripper_l_center_link`

---

## 下次运行检查清单

- [ ] `[Diag] step=1 real_gripper=[x,y,z]` 出现（确认查询成功）
- [ ] 对比 `real_gripper` z 值与 `eef_w` z 值的差距
- [ ] 对比 `real_grip-carton dist` 与 `d=` 的差距
- [ ] 如果 z 差距明显（>0.05m），根据差值调整 ROBOT_BASE z
