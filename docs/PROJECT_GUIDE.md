# GenieSim RL — Sorting Packages 项目总结文档

> **目标**: 使用 OmniReset 方法论，通过强化学习解决 AGIBOT World Challenge 2026 的 Sorting Packages 赛题。
>
> **比赛截止**: 2026-04-20（在线评测服务器关闭）

---

## 1. 项目概述

### 1.1 比赛背景

AGIBOT World Challenge 2026（ICRA 2026，奖金池 $530,000）包含 10 个操控任务。本项目针对其中的 **Sorting Packages**（包裹分拣）任务，要求 AgiBot G2 双臂机器人完成 6 步物流分拣操作：

| 步骤 | 描述 | 评分 |
|------|------|------|
| 1 | 右手末端执行器追踪目标包裹 | 0.16 |
| 2 | 用右手抓取目标包裹 | 0.16 |
| 3 | 将包裹放到扫描台上 | 0.16 |
| 4 | 包裹条码面朝上 | 0.16 |
| 5 | 再次用右手抓取包裹 | 0.16 |
| 6 | 将包裹放入目标箱子 | 0.16 |

### 1.2 技术方案：OmniReset

本项目借鉴 [OmniReset](https://arxiv.org/abs/2603.15789)（ICLR 2026）论文的核心思想：

> **多样化的模拟器重置分布——而非奖励工程或课程学习——是解决长时域接触密集操控任务的关键。**

传统 RL 训练每次都从初始状态（包裹在桌上）开始，导致策略很少练习后期阶段。OmniReset 将环境重置到任务的各个中间状态，使策略能高效练习每个阶段。

**"向后学习"效应**：PPO 自然地先学会最后阶段（near-goal 重置容易成功），然后通过 value function 向前传播，最终学会从头到尾完成整个任务。

### 1.3 六种重置分布

| 编号 | 重置类型 | 练习阶段 | 场景描述 |
|------|---------|---------|---------|
| 0 | `PackageOnTable_EEFar` | Stage 0→1 | 包裹在工作台上，末端执行器在远处 |
| 1 | `PackageOnTable_EENear` | Stage 1→2 | 包裹在工作台上，末端执行器已接近 |
| 2 | `PackageGrasped_AboveWorkspace` | Stage 2→3 | 包裹在夹爪中，正往扫描台移动 |
| 3 | `PackageOnScanTable_RandomOri` | Stage 3→4 | 包裹在扫描台上，条码朝向随机 |
| 4 | `PackageOnScanTable_BarcodeUp` | Stage 4→5 | 包裹在扫描台上，条码已朝上 |
| 5 | `PackageGrasped_NearBox` | Stage 5→6 | 包裹在夹爪中，靠近目标箱子 |

---

## 2. 项目架构

### 2.1 目录结构

```
genie_sim_RL/
├── config/                          # 环境和智能体配置
│   ├── __init__.py                  # Gymnasium 环境注册（3个环境ID）
│   ├── env_cfg.py                   # 完整环境配置（场景、动作、观测、奖励、终止、事件）
│   ├── g2_robot.py                  # G2 机器人关节定义和 USD 路径
│   └── agents/
│       ├── __init__.py
│       └── rsl_rl_cfg.py            # PPO 超参数配置（Vanilla + OmniReset 两套）
│
├── mdp/                             # 马尔可夫决策过程组件
│   ├── __init__.py                  # 统一导出所有 MDP 函数
│   ├── observations.py              # 10 个观测函数（关节、末端、包裹、目标、阶段）
│   ├── rewards.py                   # SortingStageTracker + 9 个奖励函数
│   ├── terminations.py              # 4 个终止条件
│   ├── events.py                    # 重置事件（包裹随机化、阶段推断）
│   └── reset_manager.py             # ★ OmniReset 核心：SortingMultiStageResetManager
│
├── scripts/                         # 可执行脚本
│   ├── train.py                     # 训练入口（支持 --omnireset 快捷参数）
│   ├── play.py                      # 策略评估和推理
│   └── record_sorting_reset_states.py  # ★ 重置状态数据集生成
│
├── UWLab/                           # UW Lab 框架（Isaac Lab 扩展，119MB）
│   └── ...                          # 包含 Isaac Lab、RSL-RL、OmniReset 参考实现
│
├── data/                            # 运行时生成的数据（需手动创建）
│   └── reset_states/                # 预录制的重置状态 .pt 文件
│
├── logs/                            # 训练日志和检查点（自动生成）
│
├── docs/                            # 文档
│   └── PROJECT_GUIDE.md             # 本文档
│
├── README.md                        # GenieSim 3.0 官方说明
├── requirements.txt                 # Python 依赖
├── __init__.py                      # 包标识
└── LICENSE                          # Mozilla Public License 2.0
```

### 2.2 文件详解

#### 配置层 (`config/`)

| 文件 | 行数 | 作用 |
|------|------|------|
| `__init__.py` | 29 | 注册 3 个 Gymnasium 环境 ID：`...-State-Train-v0`（Vanilla 训练）、`...-OmniReset-Train-v0`（OmniReset 训练）、`...-State-Eval-v0`（评估） |
| `env_cfg.py` | 619 | **核心配置文件**。定义仿真场景（机器人 + 3张桌子 + 包裹）、动作空间（7-DOF 手臂 + 二值夹爪）、观测空间（41D policy / 48D critic）、奖励函数组合、终止条件、训练/评估事件配置。包含 `TrainEventCfg`（Vanilla）和 `OmniResetTrainEventCfg`（OmniReset）两套事件配置 |
| `g2_robot.py` | 142 | AgiBot G2 机器人 URDF/USD 定义。包括 5-DOF 躯干（锁定）、7-DOF 左臂（锁定）、7-DOF 右臂（RL 控制）、OmniPicker 夹爪。定义关节名称、默认姿态、执行器刚度/阻尼参数 |
| `agents/rsl_rl_cfg.py` | 91 | PPO 训练超参数。两套配置：`SortingPackagesPPORunnerCfg`（Vanilla: 48步 rollout, entropy=0.008）和 `SortingPackagesOmniResetPPORunnerCfg`（OmniReset: 32步 rollout, entropy=0.01, 8个 learning epochs） |

#### MDP 层 (`mdp/`)

| 文件 | 行数 | 作用 |
|------|------|------|
| `rewards.py` | 304 | **阶段追踪器 + 奖励函数**。`SortingStageTracker`（`ManagerTermBase`）是核心，每步更新 6 阶段进度，缓存距离/方向指标。包含 `infer_stage_from_state()` 方法支持 OmniReset 中段重置后正确推断当前阶段。9个奖励函数：阶段完成(+10)、距离塑形(3个)、条码朝向(+0.3)、抓取(+1)、安全惩罚(4个) |
| `observations.py` | 117 | 10 个观测函数：右臂关节位置/速度、夹爪位置、末端位姿、包裹相对位姿、目标相对位置、条码朝上评分、阶段 one-hot 编码。Policy 观测 41D×3步历史=123D，Critic 观测 48D |
| `terminations.py` | 43 | 4 个终止条件：任务完成（6阶段全部通过）、包裹掉落（高度<0.5m）、机器人异常（NaN / 关节速度>10）、超时（30秒） |
| `events.py` | 56 | 重置事件：`randomize_package_on_table`（Vanilla 重置）和 `infer_stage_after_reset`（OmniReset 重置后阶段推断辅助函数） |
| `reset_manager.py` | 453 | **★ OmniReset 核心组件**。`SortingMultiStageResetManager` 实现 6 种重置分布，支持两种模式：Procedural（程序化生成，无需预计算）和 Dataset（加载 .pt 数据集，更高质量）。包含自适应概率调整（成功率低的阶段获得更高重置概率） |

#### 脚本层 (`scripts/`)

| 文件 | 行数 | 作用 |
|------|------|------|
| `train.py` | 120 | 训练入口。支持单 GPU / 多 GPU 分布式训练。`--omnireset` 参数快捷启用 OmniReset 模式。构建环境、加载 PPO 配置、启动 `OnPolicyRunner` 训练循环 |
| `play.py` | 98 | 策略评估脚本。加载 checkpoint，运行指定 episode 数，统计平均阶段完成数、平均得分和成功率 |
| `record_sorting_reset_states.py` | 336 | **★ 重置状态数据集生成**。在仿真中程序化生成 6 种重置状态，物理沉淀后过滤有效状态，保存为 .pt 文件。支持单类型或全部类型生成 |

---

## 3. 核心技术细节

### 3.1 机器人 — AgiBot G2

```
G2 机器人结构：
├── 躯干 (5-DOF) — 锁定（stiffness=1000, damping=100）
├── 头部 (3-DOF) — 锁定
├── 左臂 (7-DOF) — 锁定在休息姿态
├── 右臂 (7-DOF) — RL 控制（stiffness=100, damping=40）
│   └── 关节: idx61_arm_r_joint1 ~ idx67_arm_r_joint7
└── 右夹爪 (1-DOF) — RL 控制（stiffness=17, damping=5）
    └── 关节: idx81_gripper_r_outer_joint1（开=0.85, 合=0.0）
```

### 3.2 场景布局（俯视图，机器人在原点面朝 +X）

```
                    +Y
                     │
                     │
     Target Box      │        Workspace Table
     (0.3, +0.5)     │        (0.5, 0.0)
     0.4×0.4×0.3     │        0.8×1.0×0.02
                     │        ← 包裹初始位置 (0.5, 0, 1.20)
─────────────────────┼──────────────────────── +X
                     │
     Scanning Table  │
     (0.3, -0.5)     │
     0.4×0.4×0.02    │
                     │
                   Robot
                   (0, 0)
```

### 3.3 动作空间（8D）

| 维度 | 类型 | 配置 |
|------|------|------|
| 0-6 | 7-DOF 右臂关节增量 | `RelativeJointPositionActionCfg`, scale=0.05 |
| 7 | 夹爪开合（二值） | `BinaryJointPositionActionCfg`, open=0.85 / close=0.0 |

### 3.4 观测空间

**Policy 观测**（41D × 3步历史 = 123D，带噪声损坏）：

| 观测项 | 维度 | 说明 |
|--------|------|------|
| `right_arm_joints` | 7 | 右臂关节位置 |
| `gripper_pos` | 1 | 夹爪关节位置 |
| `ee_position` | 3 | 末端执行器相对机器人根部的位置 |
| `ee_orientation` | 4 | 末端执行器四元数 (wxyz) |
| `pkg_pos_rel_ee` | 3 | 包裹相对末端的位置 |
| `pkg_orientation` | 4 | 包裹四元数 |
| `scan_table_rel_ee` | 3 | 扫描台相对末端的位置 |
| `box_rel_ee` | 3 | 目标箱子相对末端的位置 |
| `pkg_rel_scan` | 3 | 包裹相对扫描台的位置 |
| `pkg_rel_box` | 3 | 包裹相对箱子的位置 |
| `barcode_up` | 1 | 条码朝上评分 [-1, 1] |
| `stage` | 6 | 当前阶段 one-hot 编码 |

**Critic 观测**（48D，不带噪声，history=1）：Policy 所有观测 + `right_arm_vel` (7D)

### 3.5 奖励设计

```
总奖励 = Σ(weight × reward_term)

稀疏奖励：
  stage_completion     × 10.0    每完成一个阶段 +1.0

密集塑形：
  ee_to_package        ×  0.5    tanh 距离奖励（阶段 0/1/4）
  package_to_scan      ×  0.5    tanh 距离奖励（阶段 2/3）
  package_to_box       ×  0.5    tanh 距离奖励（阶段 5）
  barcode_orientation  ×  0.3    条码朝上程度（阶段 3/4/5）
  grasp                ×  1.0    抓取成功（阶段 1/4）

安全惩罚：
  action_l2            × -1e-4   动作幅度惩罚
  action_rate          × -1e-3   动作变化率惩罚
  joint_vel            × -1e-3   关节速度惩罚
  package_dropped      × -5.0    包裹掉落惩罚
```

### 3.6 阶段转换条件

| 转换 | 条件 |
|------|------|
| 0→1 | 末端与包裹距离 < 8cm |
| 1→2 | 包裹高于扫描台 10cm + 夹爪关闭（pos < 0.3） |
| 2→3 | 包裹 XY 距扫描台 < 10cm + Z 距扫描台面 < 8cm |
| 3→4 | 条码朝上评分 > 0.8 + 包裹在扫描台 15cm 内 |
| 4→5 | 包裹高于扫描台 10cm + 夹爪关闭 |
| 5→6 | 包裹与箱子距离 < 12cm |

### 3.7 Domain Randomization

| 参数 | 范围 | 模式 |
|------|------|------|
| 包裹位置 X | [0.35, 0.65] | reset |
| 包裹位置 Y | [-0.3, 0.3] | reset |
| 包裹 yaw | [-π, π] | reset |
| 机器人摩擦（静/动） | [0.4,1.0] / [0.3,0.8] | startup |
| 包裹摩擦（静/动） | [0.5,1.5] / [0.4,1.2] | startup |
| 包裹质量 | [0.02, 0.15] kg | startup |

---

## 4. 使用指南

### 4.1 环境准备

**前置条件**：
- NVIDIA GPU（RTX 3090/4090/L40S 或更高，≥24GB 显存）
- NVIDIA Isaac Sim v5.1.0
- Python 3.10+
- PyTorch 2.x with CUDA

```bash
# 1. 安装 Isaac Sim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 2. 安装 UWLab（已包含在项目中）
cd genie_sim_RL/UWLab
./uwlab.sh --install

# 3. 安装项目依赖
cd ..
pip install -r requirements.txt
```

### 4.2 训练

#### 方式 A：OmniReset 训练（推荐）

```bash
# 直接训练（使用 procedural 重置，无需预计算）
./UWLab/uwlab.sh -p scripts/train.py \
    --omnireset \
    --num_envs 4096 \
    --max_iterations 40000 \
    --headless

# 或者用完整 task ID
./UWLab/uwlab.sh -p scripts/train.py \
    --task GenieSim-G2-SortingPackages-OmniReset-Train-v0 \
    --num_envs 4096 \
    --headless
```

#### 方式 B：OmniReset + 预录制数据集（更高质量）

```bash
# Step 1: 生成重置状态数据集（约 5-10 分钟）
./UWLab/uwlab.sh -p scripts/record_sorting_reset_states.py \
    --num_envs 4096 \
    --num_states 10000 \
    --headless

# Step 2: 在 config/env_cfg.py 中取消注释 dataset_dir 参数：
#   "dataset_dir": os.path.join(os.path.dirname(__file__), "..", "data", "reset_states"),

# Step 3: 训练
./UWLab/uwlab.sh -p scripts/train.py --omnireset --headless
```

#### 方式 C：Vanilla 训练（基线对比）

```bash
./UWLab/uwlab.sh -p scripts/train.py \
    --task GenieSim-G2-SortingPackages-State-Train-v0 \
    --num_envs 4096 \
    --max_iterations 50000 \
    --headless
```

#### 多 GPU 分布式训练

```bash
python -m torch.distributed.run \
    --nnodes 1 --nproc_per_node 4 \
    scripts/train.py \
    --omnireset \
    --num_envs 16384 \
    --headless \
    --distributed
```

#### 从 checkpoint 恢复训练

```bash
./UWLab/uwlab.sh -p scripts/train.py \
    --omnireset \
    --resume_path logs/g2_sorting_packages_omnireset_XXXXXXXX/model_5000.pt \
    --headless
```

### 4.3 评估

```bash
./UWLab/uwlab.sh -p scripts/play.py \
    --checkpoint logs/g2_sorting_packages_omnireset_XXXXXXXX/model_40000.pt \
    --num_envs 4 \
    --num_episodes 100

# 无头模式（纯统计）
./UWLab/uwlab.sh -p scripts/play.py \
    --checkpoint logs/.../model_40000.pt \
    --num_envs 32 \
    --num_episodes 500 \
    --headless
```

**输出示例**：

```
============================================================
  Results (500 episodes)
  Avg stages: 4.82 / 6
  Avg score:  0.771 / 1.0
  Success:    62.4%
============================================================
```

### 4.4 切换包裹变体

项目支持 4 种包裹 USD 模型（carton_020/028/029/030），可在命令行切换：

```bash
./UWLab/uwlab.sh -p scripts/train.py \
    --omnireset --headless \
    env.scene.package=carton_028
```

### 4.5 TensorBoard 监控

```bash
tensorboard --logdir logs/
```

关注的关键指标：
- `Metrics/reset_type_X_success_rate`：各重置类型的成功率
- `Metrics/reset_type_X_prob`：各重置类型的采样概率（自适应模式下会变化）
- `Reward/stage_completion`：阶段完成奖励

---

## 5. 仿真参数

| 参数 | 值 |
|------|------|
| 物理引擎 | PhysX 5 (TGS solver) |
| 物理频率 | 120 Hz |
| 策略频率 | 30 Hz（decimation=4） |
| Episode 时长 | 30s（Vanilla）/ 20s（OmniReset） |
| 并行环境数 | 4096（训练）/ 32（评估） |
| PhysX position iterations | 64 |
| 环境间距 | 3.0m |

---

## 6. PPO 超参数对比

| 参数 | Vanilla | OmniReset |
|------|---------|-----------|
| 实验名 | `g2_sorting_packages` | `g2_sorting_packages_omnireset` |
| Rollout 步数/env | 48 | 32 |
| Max iterations | 50,000 | 40,000 |
| Actor/Critic 网络 | [512,256,128,64] | [512,256,128,64] |
| 激活函数 | ELU | ELU |
| 探索噪声 | gSDE | gSDE |
| Learning rate | 1e-4 (adaptive) | 1e-4 (adaptive) |
| Entropy coef | 0.008 | **0.01** |
| Learning epochs | 5 | **8** |
| Clip param | 0.2 | 0.2 |
| GAE λ | 0.95 | 0.95 |
| Discount γ | 0.99 | 0.99 |

---

## 7. OmniReset vs Vanilla 对比

| 维度 | Vanilla | OmniReset |
|------|---------|-----------|
| 重置策略 | 始终从 stage 0 开始 | 从 6 种中间状态随机重置 |
| 后期阶段练习 | 稀少（需先通过前期阶段） | 充分（直接从该阶段开始） |
| 学习效率 | 低（大量 rollout 浪费在已会的阶段） | 高（每个阶段均匀练习） |
| 概率调整 | 无 | 自适应（弱项阶段多练习） |
| 收敛速度 | 慢 | 快（向后学习效应） |
| 预计算 | 无 | 可选（生成重置数据集） |

---

## 8. 比赛提交

比赛评测通过 HTTP 通信：仿真器发送观测图像 + 本体感受状态 → 模型返回控制命令 → 仿真器执行。

基线模型 ACoT-VLA 的推理服务可通过：

```bash
bash scripts/server.sh <GPU_ID> <PORT>
```

如需将 RL 策略转换为比赛提交格式，需要：
1. 从 RL checkpoint 导出 policy 网络
2. 包装为 HTTP inference server（接收观测、返回动作）
3. 适配比赛 API 格式

---

## 9. 参考资料

- [OmniReset 论文 (arXiv:2603.15789)](https://arxiv.org/abs/2603.15789) — 核心方法论
- [OmniReset 官网](https://weirdlabuw.github.io/omnireset/) — 可视化和补充材料
- [UWLab 框架](https://uw-lab.github.io/UWLab/) — Isaac Lab 扩展
- [GenieSim 3.0 (arXiv:2601.02078)](https://arxiv.org/abs/2601.02078) — 仿真平台
- [AGIBOT World Challenge 2026](https://agibot-world.com/challenge2026) — 比赛页面
- [ACoT-VLA 基线](https://github.com/AgibotTech/ACoT-VLA) — 比赛基线模型
