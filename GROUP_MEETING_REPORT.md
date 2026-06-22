# 🧠 老年人跌倒风险预测系统 — 组会汇报



---

## 一、项目定位

### 1.1 要解决什么问题？

老年人跌倒是居家养老中最常见也最危险的事故。**跌倒后检测** 只能用于事后报警，而本项目的目标是 **跌倒前预警**——在跌倒发生前 3~4 秒感知到姿态异常，为护工或家属争取宝贵的干预时间。

### 1.2 系统定义

```
输入: 连续视频流（RGB 摄像头）
输出: 
    ├── risk ∈ [0, 1]     ← 跌倒风险概率
    ├── time ∈ [0, ∞)     ← 距跌倒完成剩余帧数
    └── state ∈ {SAFE, WARNING, DANGER}  ← 三态报警
```

### 1.3 核心量化指标

| 指标 | 当前值 | 说明 |
|------|--------|------|
| Risk MAE（总体） | **0.0341** | 风险预测平均绝对误差 |
| Risk MAE（跌倒样本） | 0.0637 | 对跌倒样本的误差 |
| Risk MAE（正常样本） | 0.0181 | 对正常样本的误差 |
| Time MAE（跌倒样本） | **2.21 帧** | 距跌倒时间预测误差 |
| 提前预警时间 | **~3.6 秒**（90 帧 @25fps） | 跌倒前开始感知风险 |
| 数据集规模 | 7,243 个样本 | 来自 130 个视频 |

---

## 二、技术路线（7 步流水线）

```
Camera Frame (BGR, H×W×3)
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ ① PoseExtractor — YOLO11n-Pose 姿态估计                     │
│    提取画面中所有人的 17 个 COCO 关键点 (x, y)                │
│    In: frame  →  Out: [{bbox, keypoints(17,2), score}]     │
├────────────────────────────────────────────────────────────┤
│ ② PersonTracker — Centroid Tracking 多目标跟踪              │
│    基于 bbox 质心匹配，为每人分配稳定 ID                      │
│    匹配阈值 100px | 最大丢失 30 帧                           │
├────────────────────────────────────────────────────────────┤
│ ③ FeatureBuilder — 单帧 57 维特征工程 ⭐                    │
│    基础: 17×(x, y, confidence) = 51 维                     │
│    几何: 重心高度 + 躯干倾斜角 + 左右膝角 + 双脚距离 + 体高 = 6 维│
├────────────────────────────────────────────────────────────┤
│ ④ SequenceBufferV3 — 30 帧滑动窗口缓存                      │
│    维护 deque(maxlen=30)，同时计算速度、加速度                │
├────────────────────────────────────────────────────────────┤
│ ⑤ 运动特征拼接 — (30, 171)                                  │
│    [pos(30,57) | vel(30,57) | acc(30,57)]                  │
├────────────────────────────────────────────────────────────┤
│ ⑥ LSTM 多任务模型 — 双头输出                                 │
│    2 层 LSTM (hidden=128) → Mean Pooling →                  │
│    ├── risk_head:  Linear → Sigmoid → risk ∈ [0,1]         │
│    └── time_head:  Linear → ReLU    → time ≥ 0             │
├────────────────────────────────────────────────────────────┤
│ ⑦ RiskEngine — 风险评估引擎                                 │
│    EMA 平滑 (α=0.6) → 连续触发计数 → 三态状态机              │
│    输出: SAFE / WARNING / DANGER                            │
└────────────────────────────────────────────────────────────┘
```

### 技术栈一览

| 组件 | 技术 |
|------|------|
| 姿态估计 | Ultralytics YOLO11n-Pose (6.3 MB) |
| 时序模型 | PyTorch LSTM, 2 层, hidden=128 (1.1 MB) |
| 多目标跟踪 | Centroid Tracking（质心匹配） |
| 图像处理 | OpenCV |
| 数据处理 | NumPy, scikit-learn |
| 环境管理 | Conda (yolo 环境) |

---

## 三、核心创新点

### 3.1 57 维几何增强特征（非简单坐标展平）

仅用 17 个关键点的 (x,y) 坐标（34 维）无法区分"站立不动"和"正在跌倒但此刻位置正常"。

**每帧 57 维 = 51 基础 + 6 几何：**

| 特征 | 维度 | 跌倒时的表现 | 为什么重要 |
|------|------|-------------|-----------|
| 关键点坐标 | 17×2=34 | — | 位置信息 |
| 关键点置信度 | 17×1=17 | 遮挡时下降 | 数据质量信号 |
| **重心高度** | 1 | 快速下降 ↓ | 跌倒最直接的全局信号 |
| **躯干倾斜角** | 1 | 大幅偏离垂直 | 失去平衡的核心指标 |
| **左膝角** | 1 | 急剧弯曲/伸直 | 下肢支撑失效 |
| **右膝角** | 1 | 急剧弯曲/伸直 | 下肢支撑失效 |
| **双脚距离** | 1 | 异常变化 | 支撑面宽度骤变 |
| **人体高度** | 1 | 快速缩小 | 身体倒地的全局特征 |
| **合计** | **57** | | |

### 3.2 171 维运动序列（位置 + 速度 + 加速度）

```
序列 (30, 171) = [ pos(30,57) | vel(30,57) | acc(30,57) ]

位置: "在哪里"       → 静态姿态
速度: "往哪走"       → 运动方向/速率（跌倒时垂直速度骤增）
加速度: "趋势变化"    → 运动突变（跌倒瞬间加速度峰值，最关键的区别信号）
```

新增的 6 个几何特征的**速度**和**加速度**也被纳入模型，例如躯干倾斜角的加速度能捕捉"从直立到倾倒"的瞬间变化。

### 3.3 渐进式风险标签（非 0/1 二值）

```
risk = 1 - time / MAX_TIME    （线性衰减）

距跌倒 60 帧 (2.4s):  risk ≈ 0.0  → 安全
距跌倒 30 帧 (1.2s):  risk ≈ 0.5  → 预警中
距跌倒 10 帧 (0.4s):  risk ≈ 0.83 → 高度危险
跌倒瞬间:            risk = 1.0   → 确认跌倒
```

**优势**: 模型输出连续概率曲线，天然支持渐进式预警，而非二分类的跳变输出。

### 3.4 多任务学习（risk + time 双头）

两个任务共享同一个 LSTM 时序编码器，互相促进：

| 头部 | 任务 | Loss | 输出 |
|------|------|------|------|
| risk_head | 回归 | BCEWithLogitsLoss | 跌倒概率 |
| time_head | 回归 | SmoothL1Loss | 距跌倒帧数 |

**关键设计**: time loss 只对 fall 样本计算（normal 样本的 time=999 是占位值，无意义）。

### 3.5 三阶段风险评估（避免抖动）

```
阶段 1 — EMA 平滑:
    smooth[t] = 0.6 × prob + 0.4 × smooth[t-1]

阶段 2 — 连续触发计数:
    if smooth ≥ 0.6: counter += 1
    else:             counter = 0

阶段 3 — 状态机:
    counter ≥ 5  →  DANGER   （确认跌倒，需连续5帧高危）
    smooth ≥ 0.35 → WARNING  （跌倒预警）
    其他          →  SAFE     （安全）
```

---

## 四、数据集 & 训练管线

### 4.1 数据来源

**Le2i 跌倒数据集**：130 个监控视频，4 个场景

| 场景 | 视频数 | 内容 |
|------|--------|------|
| Coffee_room_01 | 48 | 咖啡室视角 1 |
| Coffee_room_02 | 22 | 咖啡室视角 2 |
| Home_01 | 30 | 居家场景 1 |
| Home_02 | 30 | 居家场景 2 |

每个视频有 frame-level 标注：fall_start（跌倒开始帧）、fall_end（跌倒结束帧）。

### 4.2 数据集构建流程

```
Le2i 原始视频 (.avi)
    │
    ▼
YOLO11n-Pose 逐帧提取 → keypoints(T, 17, 2) + confs(T, 17)
    │                    缓存到 yolo_keypoints/ 目录（可复用）
    ▼
FeatureBuilder 逐帧构建 57 维特征
    │
    ▼
计算速度 + 加速度 → 拼接 [pos, vel, acc] → features(T, 171)
    │
    ▼
滑动窗口切分 (window=30, stride=5)
    │
    ▼
计算 risk / time 标签
    │
    ▼
保存 x.npy, risk.npy, time.npy → D:\my_datasets\Le2i\processed\
```

### 4.3 标签计算

| 帧区间 | risk | time | 类别 |
|--------|------|------|------|
| 末帧 < fall_start − 120 | 0 | 999 | normal |
| fall_start−120 ≤ 末帧 < fall_end | 1.0 − time/60 | fall_end−1−末帧 | fall_process |
| 末帧 ≥ fall_end | 1.0 | 0 | fall_after |

### 4.4 数据分布

| 类别 | 样本数 | 占比 |
|------|--------|------|
| Normal | ~4,700+ | ~65% |
| Fall Process | ~900+ | ~13% |
| Fall After | ~1,500+ | ~21% |
| **总计** | **~7,243** | 100% |

> 样本不均衡通过风险回归的连续标签设计自然缓解——fall_process 样本的 risk 是渐进值而非全 1。

### 4.5 训练配置

| 参数 | V3 (train_lstm.py) | V3.1 (train_risk_lstm.py) |
|------|--------------------|-----------------------------|
| Epochs | 50 | 30 |
| Batch Size | 128 | 32 |
| Learning Rate | 1e-3 | 1e-3 |
| Optimizer | Adam | Adam (weight_decay=1e-5) |
| Scheduler | CosineAnnealingLR | ReduceLROnPlateau |
| Loss (risk) | MSE | BCEWithLogitsLoss |
| Loss (time) | SmoothL1Loss | SmoothL1Loss |
| λ_time | 0.1 | 0.5 |
| Gradient Clip | — | max_norm=5.0 |
| 归一化 | Z-score (train set) | Z-score (train set) |

### 4.6 训练结果（V3）

| 指标 | 值 |
|------|-----|
| **Risk MAE（总体）** | **0.0341** |
| Risk MAE（fall 样本） | 0.0637 |
| Risk MAE（normal 样本） | 0.0181 |
| **Time MAE（fall 样本）** | **2.21 帧** |

---

## 五、模型架构详图

### 5.1 推理模型：LSTMModel (V3)

```
Input: (batch, 30, 171)
    │
    ▼
LSTM Layer 1 (171→128, dropout=0.2)
    │
    ▼
LSTM Layer 2 (128→128, dropout=0.2)
    │
    ▼
Take last timestep → (batch, 128)
    │
    ├── risk_head:  Linear(128→1) → Sigmoid → risk ∈ [0, 1]
    │
    └── time_head:  Linear(128→1) → ReLU    → time ∈ [0, ∞)
```

### 5.2 训练模型：RiskLSTM (V3.1)

```
Input: (batch, 30, 171)
    │
    ▼
LSTM (171→128, 2层, dropout=0.2)
    │
    ▼
Mean Pooling over 30 timesteps → (batch, 128)
    │
    ▼
Feature Projection: Linear(128→64) → ReLU → LayerNorm(64) → Dropout(0.2)
    │
    ├── risk_head:  Linear(64→32) → ReLU → Linear(32→1) → logits (BCEWithLogits)
    │
    └── time_head:  Linear(64→32) → ReLU → Linear(32→1) → regression (SmoothL1)
```

**V3.1 相比 V3 的改进**：
- Last-step → **Mean Pooling**（更稳定，不依赖最后一帧噪声）
- 新增 **Feature Projection** 层（128→64，加 LayerNorm 稳定训练）
- risk head 加了隐藏层（64→32→1 vs 128→1）
- risk loss 从 MSE → **BCEWithLogitsLoss**（更符合分类本质）

---

## 六、实时演示

### 运行方式

```bash
conda activate yolo
cd D:\myproject\elder

# 检测视频文件
python realtime_detect.py "test/video (1).avi"

# 检测摄像头
python realtime_detect.py
```

### 可视化效果

| 状态 | 颜色 | risk 阈值 | 含义 |
|------|------|-----------|------|
| SAFE | 🟢 绿色 | risk < 0.50 | 正常 |
| WARNING | 🟠 橙色 | 0.50 ≤ risk < 0.80 | 注意 |
| FALL | 🔴 红色 | risk ≥ 0.80 | 危险 |

画面显示：边界框 + 人员 ID + 状态 + risk 值 + time 值。

---

## 七、版本演进

| 版本 | 本质 | 特征维度 | 输出 | 状态 |
|------|------|----------|------|------|
| V1/V2 | 动作分类 | 34 维（仅坐标） | SAFE / FALL 二分类 | 归档 |
| **V3** | **风险回归** | **171 维（57×3）** | **risk + time 双头** | ✅ 已训练，可运行 |
| V3.1 | 时间事件预测（工业级） | 171 维 + 增强训练 | risk + time（改进 Loss + 架构） | 模型定义完成，待训练 |

**本质变化**: 从"跌倒后识别" → "跌倒前预警"。V3 已经实现了核心能力——在跌倒前 ~3.6 秒渐进式感知风险。

---

## 八、项目工程结构

```
elder/
├── pose/                  # 姿态提取 — YOLO11n-Pose, 17 COCO 关键点
│   └── pose_extractor.py
├── tracker/               # 多目标跟踪 — Centroid Tracking
│   ├── track.py           #   单目标 Track 数据结构
│   └── person_tracker.py  #   多目标匹配器
├── features/              # 特征工程 ⭐
│   ├── feature_builder.py     #   57 维/帧（51 基础 + 6 几何）
│   └── label_generator_v3.py  #   标签生成器（risk + time）
├── sequence/              # 序列缓冲 — 30 帧 deque
│   └── sequence_buffer.py
├── classifier/            # LSTM 分类器
│   ├── lstm_classifier.py         #   推理器（LSTMModel, 171 维输入）
│   └── classifier/
│       └── risk_lstm_multitask.py #   V3.1 模型定义（RiskLSTM）
├── pipeline/              # 流水线
│   ├── fall_detector.py   #   主流水线（串联所有模块）
│   └── risk_engine.py     #   风险评估引擎（EMA + 状态机）
├── models/                # 模型权重
│   ├── yolo11n-pose.pt    #   YOLO11n 姿态估计 (6.3 MB)
│   ├── lstm_multitask.pt  #   LSTM 多任务模型 (1.1 MB)
│   └── norm_params.npz    #   Z-score 归一化参数
├── build_dataset_yolo.py  # 数据集构建（YOLO 提取 keypoints → 171 维特征）
├── train_lstm.py          # V3 训练脚本
├── realtime_detect.py     # 实时检测演示（当前主入口）
├── diagnose.py            # 诊断脚本（逐帧打印 risk/time/label）
└── scripts/               # 测试 & 训练脚本
    ├── train_risk_lstm.py     # V3.1 训练脚本
    ├── test_fall_detector.py  # 集成测试
    └── ...
```

---

## 九、已知问题 & 后续方向

| 问题 | 当前状态 | 改进方向 | 优先级 |
|------|----------|----------|--------|
| Centroid Tracker ID 漂移 | 交叉时 ID 可能互换 | DeepSORT / ByteTrack | 高 |
| 当前仅单人检测 | realtime_detect.py 固定 person_id=0 | 接入 FallDetector 多人管线 | 高 |
| V3.1 模型未训练 | 仅完成模型定义 | 用 RiskLSTM + Le2IDataset 训练 | 高 |
| LSTM 感受野有限 | 30 帧 ≈ 1.2 秒 | Transformer / TCN 扩大时序视野 | 中 |
| 配置硬编码 | 参数分散在各模块 | 统一 configs/ 配置文件 | 中 |
| 无 API 服务 | 仅本地演示 | FastAPI 封装 REST 接口 | 低 |
| 缺少 requirements.txt | 环境复现依赖手动 | 生成依赖清单 | 低 |

---

## 十、一句话总结

> **摄像头 → YOLO 17 个骨架关键点 → 57 维几何增强特征（含重心/倾斜角/膝角） → 30 帧运动序列（位置+速度+加速度, 171 维） → 2 层 LSTM 双头预测（risk + time） → EMA 平滑状态机，在跌倒前约 3.6 秒实现渐进式风险预警，Risk MAE 低至 0.034。**



1. 输出时间， 图片
2. 时序模型推风险，time
3. 调研
