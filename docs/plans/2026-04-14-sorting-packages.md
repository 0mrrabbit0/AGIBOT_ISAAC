# Sorting Packages - 实施计划

**日期**: 2026-04-14
**截止**: 2026-04-20
**状态**: 规划中
**任务**: AGIBOT Challenge 2026 - sorting_packages

---

## 1. 任务概述

控制 G2 双臂人形机器人完成包裹分拣：抓取目标包裹 → 放到扫码台（条码朝上）→ 再抓起 → 放入蓝色分拣箱 → 转腰回初始位。

**评测序列**（来自 `problems.json`）:

| # | 评测动作 | 成功条件 | 超时(步) |
|---|---------|---------|---------|
| 1 | Follow | 右夹爪进入目标包裹 bbox (0.2m) | 600 |
| 2 | PickUpOnGripper | 包裹 z 轴抬升 > 0.02m | 300 |
| 3 | Inside | 包裹中心进入扫码台 AABB (scale=0.8) | 900 |
| 4 | Upright | 包裹倾斜角 < 5° | (与3并行) |
| 5 | PickUpOnGripper | 再次抬升 > 0.02m | 900 |
| 6 | Inside | 包裹中心进入蓝色箱 AABB (scale=0.8) | 900 |

**总步数上限**: ~3600 步
**E2E 成功**: 第 6 步 score=1.0 时才算通过

---

## 2. 关键坐标与常量

### 2.1 场景物体（instance 0）

| 物体 | Prim Path | 已知位置 (scene_info) |
|------|-----------|---------------------|
| 目标包裹 (黄色) | benchmark_carton_00364e8c | [0.017, 0.613, 1.19] |
| 扫码台 | /World/background/benchmark_scanner_000 | 运行时查询 |
| 蓝色分拣箱 | /World/background/benchmark_material_tray_000 | 运行时查询 |

> **注意**: scene_info 的坐标和 USD 实际加载坐标可能有差异，所有位置都应在运行时通过 `api_core.get_obj_world_pose()` 查询。

### 2.2 机器人常量

```
ROBOT_BASE = [0.24469, 0.09325, 0.0]
INIT_BODY  = [1.57, 0.0, -0.319, 1.344, -1.045]  # bj1-bj5
INIT_RIGHT_ARM = [-0.739, -0.717, 1.524, -1.538, -0.278, -0.926, 0.839]
```

### 2.3 腰部旋转角度（bj5）

bj5 控制水平转向：
- **面向桌子** (初始): bj5 ≈ -1.045
- **面向扫码台** (右转): 需运行时根据扫码台位置计算
- **面向蓝色箱** (继续右转): 需运行时根据箱子位置计算

---

## 3. 技术方案

### 3.1 方案选择：PiEnv + IKFKSolver

**使用 PiEnv（关节空间）**，手动调用 IKFKSolver 做 IK 求解。

理由：
- run_sorting_benchmark.py 已经基于 PiEnv 搭建
- 更直接的关节控制，便于调试
- AbsPoseEnv 需要更多框架改动

### 3.2 核心控制流

```
每一步 step:
1. obs = env.get_observation()
   → obs["eef"]["right"] = [x,y,z,qw,qx,qy,qz]  (arm_base_link 坐标系)
   → obs["states"][0:14] = 当前关节角
2. 根据状态机阶段，计算目标 EEF 位姿 (世界坐标)
3. 世界坐标 → arm_base_link 局部坐标
4. IKFKSolver.eef_actions_to_joint() → 目标关节角
5. 填入 action[0:16] + action[20] (bj5)
6. env.step(action)
```

### 3.3 坐标系变换

```
世界坐标 → arm_base_link 局部坐标:
  local_pos = world_pos - ROBOT_BASE - waist_offset(bj1-bj5)
```

FK 输出在 `arm_base_link` 坐标系。需要理解 arm_base_link 相对于世界坐标系的变换关系。这个变换受腰部关节（bj1-bj5）影响。

**关键**: 由于 bj1-bj4 被我们 hold 在初始值，只有 bj5 变化，变换简化为绕 z 轴旋转 + 固定偏移。

---

## 4. 状态机设计（6 阶段）

### Phase 1: APPROACH — 接近目标包裹

**目标**: 右手末端移动到包裹上方（预抓取位）

```
输入: 包裹世界坐标 carton_pos (运行时查询)
目标: EEF → carton_pos + [0, 0, +0.10]  (上方 10cm)
腰部: 保持初始 bj5
夹爪: 张开 (action=0)
完成条件: EEF 距目标 < 0.03m
评测触发: Follow (夹爪进入包裹 bbox)
```

### Phase 2: GRASP — 抓取包裹

**目标**: 下降到抓取位，闭合夹爪

```
步骤:
  2a. EEF 下降到 carton_pos + [0, 0, +0.02]  (包裹顶部)
  2b. 闭合夹爪 (action=1), 保持 30 步
  2c. 抬升到 carton_pos + [0, 0, +0.15]  (提起)
完成条件: 包裹 z > 初始 z + 0.02m
评测触发: PickUpOnGripper
```

### Phase 3: MOVE_TO_SCANNER — 转腰到扫码台

**目标**: 转腰面向扫码台，将包裹放下

```
步骤:
  3a. 计算扫码台方向对应的 bj5 角度
  3b. 平滑插值 bj5 从初始值 → 目标值
  3c. 同时保持手臂抬起 (防止碰撞)
  3d. 到达后下降 EEF 到扫码台表面
  3e. 松开夹爪 (action=0)
完成条件: 包裹在扫码台 AABB 内 且 倾斜 < 5°
评测触发: Inside + Upright
```

**条码朝上策略**:
- 从上方竖直抓取 → 包裹不翻转
- 竖直放下 → 保持原始顶面朝上
- 如果条码不在顶面，需要调整 EEF 旋转 (roll/pitch)
- **TODO**: 需目视确认条码位置，或查看 carton USD 模型

### Phase 4: REGRASP — 从扫码台重新抓取

**目标**: 再次抓取扫码台上的包裹

```
步骤:
  4a. EEF 移到包裹上方 (此时包裹在扫码台上)
  4b. 下降，闭合夹爪
  4c. 抬升
完成条件: 包裹 z > 扫码台面 + 0.02m
评测触发: PickUpOnGripper
```

### Phase 5: MOVE_TO_BIN — 转腰到蓝色箱

**目标**: 转腰面向蓝色分拣箱，放入包裹

```
步骤:
  5a. 计算蓝色箱方向对应的 bj5 角度
  5b. 平滑插值 bj5
  5c. 到达后下降/释放包裹到箱内
  5d. 松开夹爪
完成条件: 包裹在蓝色箱 AABB 内
评测触发: Inside
```

### Phase 6: RETURN — 转腰回初始位

**目标**: 腰部回到初始朝向

```
步骤:
  6a. 平滑插值 bj5 回初始值 (-1.045)
  6b. 手臂回到 INIT_RIGHT_ARM
完成条件: bj5 ≈ 初始值
```

> 注意: 评测只看前 6 个动作（Follow → Inside），Phase 6 不被评分但任务描述要求做。

---

## 5. 待解决问题

### P0 — 必须解决

| # | 问题 | 解决方式 |
|---|------|---------|
| 1 | arm_base_link 相对世界坐标系的精确变换 | 运行时查 FK + 对比已知位置 |
| 2 | IKFKSolver 初始化方式 (在 policy 中能否直接用) | 查 PiEnv 如何初始化的，复用 |
| 3 | bj5 角度与水平朝向的映射关系 | 用 atan2 从目标位置计算 |
| 4 | 夹爪开合对应的 action 值 | action=0 开, action=1 合 (经 relabel) |

### P1 — 需要确认

| # | 问题 | 解决方式 |
|---|------|---------|
| 5 | 条码在包裹哪个面 | 目视确认或查 USD 模型贴图 |
| 6 | 扫码台/蓝色箱的精确位置和尺寸 | 运行时 get_obj_world_pose 查询 |
| 7 | instance 1 (黑色包裹) 的差异 | 仅目标 carton_id 不同，逻辑相同 |

### P2 — 可以后做

| # | 问题 | 说明 |
|---|------|-----|
| 8 | 碰撞避让（转腰时手臂可能碰桌子） | 转腰时先抬高手臂 |
| 9 | 运动平滑性 | 关节插值 alpha 调参 |
| 10 | 其他 9 个任务的实现 | sorting_packages 通过后再做 |

---

## 6. 实施步骤

### Step 1: 环境验证（~30 min）
- [ ] 确认 bj1-bj4 hold patch 生效，手臂不再抬高
- [ ] 运行时打印: 包裹、扫码台、蓝色箱的世界坐标
- [ ] 运行时打印: 当前 EEF 位置 (obs["eef"]["right"])
- [ ] 确认 IKFKSolver 可在 policy 中使用

### Step 2: 坐标系标定（~1 hr）
- [ ] 对比 FK 输出的 EEF 位置与 get_obj_world_pose 的差异
- [ ] 确定 arm_base_link → 世界坐标的变换公式
- [ ] 验证: 给定包裹世界坐标 → IK → 关节角 → FK → 回到原坐标

### Step 3: 实现 Phase 1-2 抓取（~2 hr）
- [ ] 实现 APPROACH: EEF 移动到包裹上方
- [ ] 实现 GRASP: 下降 + 闭合 + 抬起
- [ ] 验证 Follow 和 PickUpOnGripper 评测通过

### Step 4: 实现 Phase 3 扫码台放置（~2 hr）
- [ ] 实现腰部旋转到扫码台
- [ ] 实现放置动作
- [ ] 处理条码朝上 (Upright 评测)
- [ ] 验证 Inside + Upright 评测通过

### Step 5: 实现 Phase 4-5 重新抓取+放入箱子（~2 hr）
- [ ] 实现从扫码台重新抓取
- [ ] 实现腰部旋转到蓝色箱
- [ ] 实现放入箱子
- [ ] 验证第二次 PickUpOnGripper + Inside 评测通过

### Step 6: 实现 Phase 6 + 端到端测试（~1 hr）
- [ ] 实现转腰回初始位
- [ ] 端到端运行，确认 E2E=1
- [ ] 测试 instance 1 (黑色包裹)

---

## 7. 文件结构

```
scripts/
├── scripted_sorting_policy.py   # 状态机策略 (重写)
├── run_sorting_benchmark.py     # 启动脚本 (已有, 小改)
└── setup_env.sh                 # 环境部署 (已有)
```

所有逻辑集中在 `scripted_sorting_policy.py`，不修改 genie_sim 框架源码（仅通过 monkey-patch）。
