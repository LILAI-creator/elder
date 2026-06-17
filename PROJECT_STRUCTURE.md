# 🧠 Elder Fall Risk Prediction System — 项目结构文档

> **版本**: v3.1 | **日期**: 2026-06-15 | **分支**: master

---

## 1. 项目概述

基于 **YOLO11n-Pose 姿态估计 + LSTM 时序深度学习** 的实时老年人跌倒风险预测系统。从摄像头视频流中提取人体 17 个 COCO 关键点，通过滑动窗口构建 171 维时序特征（57 维/帧 × 3：位置+速度+加速度），由多任务 LSTM 同时预测 **跌倒风险概率（risk）** 和 **距跌倒完成时间（time）**，最终经 RiskEngine 输出 SAFE / WARNING / DANGER 三态报警。

### 1.1 核心能力

| 能力 | 说明 |
|------|------|
| 实时姿态提取 | YOLO11n-Pose，17 个 COCO 关键点，单帧推理 |
| 多目标跟踪 | Centroid Tracking，为每人分配固定 ID |
| 时序动作分析 | 2 层 LSTM，30 帧滑动窗口，171 维特征输入 |
| 多任务预测 | 双头输出：risk（跌倒概率）+ time（距跌倒帧数） |
| 风险评估 | EMA 平滑 + 连续触发计数 + 三态状态机 |
| 渐进式预警 | 跌倒前约 3.6 秒（90 帧 @25fps）开始感知风险 |

### 1.2 技术栈

| 组件 | 技术 |
|------|------|
| 姿态估计 | Ultralytics YOLO11n-Pose |
| 时序模型 | PyTorch LSTM（2层，hidden=128） |
| 多目标跟踪 | Centroid Tracking（质心匹配） |
| 图像处理 | OpenCV |
| 数据处理 | NumPy, scikit-learn |
| 环境管理 | Conda（yolo 环境） |

---

## 2. 项目目录结构

```
D:\myproject\elder\
│
├── pose/                               # 🎯 姿态提取模块
│   ├── __init__.py
│   └── pose_extractor.py               # YOLO11n-Pose 关键点提取器
│
├── tracker/                            # 👥 多目标跟踪模块
│   ├── __init__.py
│   ├── track.py                        # 单目标 Track 数据结构
│   └── person_tracker.py               # Centroid 多目标跟踪器
│
├── features/                           # 📐 特征工程模块
│   ├── __init__.py
│   ├── feature_builder.py              # V3 特征构建器（51维：坐标+置信度）
│   └── label_generator_v3.py           # V3 标签生成器（risk + time_to_fall）
│
├── sequence/                           # 📦 序列缓冲模块
│   ├── __init__.py
│   └── sequence_buffer.py              # 30帧 deque 缓存器（含速度/加速度计算）
│
├── classifier/                         # 🧠 分类器模块
│   ├── lstm_classifier.py              # LSTM 推理器（102维输入，risk+time 双头）
│   └── classifier/
│       └── risk_lstm_multitask.py      # V3.1 多任务 LSTM 模型定义（RiskLSTM）
│
├── pipeline/                           # 🔄 流水线模块
│   ├── __init__.py
│   ├── fall_detector.py                # 主流水线（串联所有模块）
│   └── risk_engine.py                  # 风险评估引擎（EMA + 连续触发 + 状态机）
│
├── models/                             # 💾 训练好的模型权重
│   ├── yolo11n-pose.pt                 # YOLO11n 姿态估计（6.3 MB）
│   ├── lstm_multitask.pt               # LSTM 多任务模型（v2，1 MB）
│   ├── lstm_multitask1.pt              # LSTM 多任务模型（v1，1 MB）
│   ├── lstm_baseline.pt                # LSTM 基线模型（868 KB）
│   └── norm_params.npz                 # Z-score 归一化参数（mean, std）
│
├── configs/                            # ⚙️ 配置（预留，尚未实现）
│
├── dataset/                            # 📊 数据集加载器
│   └── le2i_dataset.py                 # Le2i 数据集 PyTorch Dataset
│
├── scripts/                            # 🔧 测试 & 训练脚本
│   ├── build_dataset.py                # 构建数据集入口
│   ├── build_le2i_dataset.py           # Le2i 数据集预处理
│   ├── train_risk_lstm.py              # LSTM 模型训练脚本（V3.1）
│   ├── realtime_demo.py                # 流水线版实时演示
│   ├── test_pose_extractor.py          # PoseExtractor 单元测试
│   ├── test_pose_v3.py                 # PoseExtractor V3 测试
│   ├── test_tracker.py                 # PersonTracker 单元测试
│   ├── test_fall_detector.py           # FallDetector 集成测试
│   ├── test_lstm_classifier.py         # LSTM 分类器测试
│   ├── test_sequence_buffer.py         # SequenceBuffer 测试
│   └── test_label_generator_v3.py      # LabelGenerator 测试
│
├── test/                               # 🎬 测试视频
│   ├── video (1).avi                   # 36 MB 测试视频
│   ├── video (2).avi                   # 71 MB 测试视频
│   └── video (22).avi                  # 53 MB 测试视频
│
├── build_dataset.py                    # 数据集构建（从预提取 keypoints）
├── build_dataset_yolo.py               # 数据集构建（用 YOLO 重新提取 keypoints）
├── train_lstm.py                       # LSTM 多任务模型训练（V3）
├── realtime_detect.py                  # 实时检测演示（当前主入口）
├── realtime_demo.py                    # 旧版实时演示（命令行参数版）
├── diagnose.py                         # 诊断脚本（逐帧打印 risk/time/label）
├── fall_predict.py                     # 早期预测模块（辅助函数集合）
├── inspect_keypoints.py                # 检查预提取 keypoints 数据
├── check_annotations.py                # 检查 Le2i 标注文件
├── test_pose.py                        # YOLO-Pose 摄像头快速测试
│
├── PROJECT_DOC.md                      # 详细技术文档（V3）
├── PROJECT_STRUCTURE.md                # 📍 本文件 — 项目结构文档
├── v3.md                               # V3 设计文档
├── v3.1.md                             # V3.1 设计文档
├── TDD-Elder-Fall-Detection-*.md       # TDD 开发文档
├── xiangmu.md                          # 项目概要
├── readme.md                           # 快速使用说明
└── image.png                           # 测试图片
```

---

## 3. 系统架构 — 数据流

```
Camera Frame (BGR, H×W×3)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ 1. PoseExtractor                   pose/pose_extractor.py
│    YOLO11n-Pose → 17 COCO 关键点 per person
│    In:  frame (ndarray)
│    Out: [{bbox(4,), keypoints(17,2), score}]
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│ 2. PersonTracker                 tracker/person_tracker.py
│    Centroid Tracking → 稳定 ID 分配
│    In:  [{bbox, keypoints, score}]
│    Out: (tracks, removed_ids)
│    - 匹配阈值: 100px | 最大丢失: 30帧
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│ 3. FeatureBuilder              features/feature_builder.py
│    keypoints(17,2) + confidence(17,) → feature(57,)
│    51维基础(17×3) + 6维几何特征
│    几何: 重心高度, 躯干倾斜角, 左/右膝角, 双脚距离, 体高
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│ 4. SequenceBufferV3            sequence/sequence_buffer.py
│    30帧 deque 缓存 + 速度/加速度计算
│    In:  (person_id, feature(57,))
│    Out: raw(30,57), vel(30,57), acc(30,57)
│    - 自动过滤 NaN 和 shape 异常
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│ 5. Feature Concatenation            realtime_detect.py
│    [pos(30,57) | vel(30,57) | acc(30,57)]
│    → sequence(30,171)
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│ 6. LSTMClassifier              classifier/lstm_classifier.py
│    2层 LSTM → 双头输出
│    In:  sequence(30,171) + Z-score 归一化
│    Out: {risk: float, time: float, label: int}
│    Model: LSTMModel (hidden=128, dropout=0.2)
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│ 7. RiskEngine                    pipeline/risk_engine.py
│    EMA平滑 → 连续触发计数 → 三态状态机
│    In:  (person_id, danger_prob)
│    Out: {risk_score: float, state: SAFE/WARNING/DANGER}
│    - α=0.6 | danger_thres=0.6 | trigger_frames=5
└──────────────────────────────────────────────────────┘
```

---

## 4. 核心模块详解

### 4.1 PoseExtractor — 姿态提取

**文件**: `pose/pose_extractor.py` | **类**: `PoseExtractor`

| 属性 | 值 |
|------|-----|
| 模型 | YOLO11n-Pose |
| 关键点 | 17 个 COCO 格式 (x, y) |
| 坐标系 | 绝对像素坐标，原点在左上角 |

**17 个 COCO 关键点**:

| # | 部位 | # | 部位 | # | 部位 |
|---|------|---|------|---|------|
| 0 | 鼻子 | 6 | 右肩 | 12 | 右髋 |
| 1 | 左眼 | 7 | 左肘 | 13 | 左膝 |
| 2 | 右眼 | 8 | 右肘 | 14 | 右膝 |
| 3 | 左耳 | 9 | 左腕 | 15 | 左踝 |
| 4 | 右耳 | 10 | 右腕 | 16 | 右踝 |
| 5 | 左肩 | 11 | 左髋 | | |

**API**:

```python
extractor = PoseExtractor(model_path="./models/yolo11n-pose.pt")
persons = extractor.extract(frame)
# → [{"bbox": (4,), "keypoints": (17,2), "score": float}, ...]

feature = PoseExtractor.build_feature(person)
# → (34,) = keypoints.reshape(-1)
```

### 4.2 PersonTracker — 多目标跟踪

**文件**: `tracker/person_tracker.py` | **类**: `PersonTracker`
**数据结构**: `tracker/track.py` | **类**: `Track`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_missing` | 30 | 最大允许丢失帧数 |
| `distance_threshold` | 100 | 质心匹配距离阈值（像素） |

**算法流程**:
1. 计算当前帧每人 bbox 中心点
2. 与已有 Track 的 bbox 中心点计算欧氏距离
3. 距离 < 阈值 → 匹配成功，更新 Track
4. 未匹配的检测 → 创建新 Track，分配新 ID
5. 连续丢失 > max_missing → 删除 Track，返回 removed_ids

**Track 数据结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 唯一标识 |
| `bbox` | ndarray(4,) | 边界框 [x1,y1,x2,y2] |
| `keypoints` | ndarray(17,2) | 最新关键点坐标 |
| `missing` | int | 连续丢失帧数 |
| `age` | int | 存活总帧数 |
| `hit_count` | int | 成功匹配次数 |

### 4.3 FeatureBuilder — 特征工程

**文件**: `features/feature_builder.py` | **类**: `FeatureBuilder`

| 版本 | 维度 | 构成 |
|------|------|------|
| 当前 | 57 | 17 关键点 × (2 坐标 + 1 置信度) = 51 + 6 几何特征 |

**6 个几何特征**:

| 特征 | 计算方式 | 跌倒时的表现 |
|------|----------|-------------|
| 重心高度 | 髋部中点 Y 坐标 | 快速下降 |
| 躯干倾斜角 | arctan2(肩髋连线) | 大幅偏离垂直 |
| 左膝角 | 左髋-左膝-左踝夹角 | 急剧弯曲/伸直 |
| 右膝角 | 右髋-右膝-右踝夹角 | 急剧弯曲/伸直 |
| 双脚距离 | 左右脚踝欧氏距离 | 异常变化 |
| 人体高度 | 鼻子到脚踝中心距离 | 快速缩小 |

```python
builder = FeatureBuilder()
feature = builder.build(kpts=(17,2), confs=(17,))
# → (57,) = [x0,y0,c0, x1,y1,c1, ..., x16,y16,c16, cg_y, torso_angle, l_knee, r_knee, foot_dist, body_h]
```

### 4.4 SequenceBufferV3 — 序列缓冲

**文件**: `sequence/sequence_buffer.py` | **类**: `SequenceBufferV3`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `seq_len` | 30 | 滑动窗口帧数 |

**核心方法**:

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `update(person_id, feature)` | feature(57,) | — | 追加一帧到缓冲区 |
| `is_ready(person_id)` | — | bool | 缓冲区是否已满 30 帧 |
| `get_sequence(person_id)` | — | (30,57) | 获取原始位置序列 |
| `get_velocity(person_id)` | — | (30,57) | v[t] = pos[t] − pos[t−1] |
| `get_acceleration(person_id)` | — | (30,57) | a[t] = vel[t] − vel[t−1] |
| `remove(person_id)` | — | — | 清理消失人物 |

**安全保护**: 自动过滤 NaN 值、shape 不匹配的异常特征。

### 4.5 171 维特征拼接

**位置**: `realtime_detect.py` → `build_motion_feature()`

```
sequence(30,171) = [ pos(30,57) | vel(30,57) | acc(30,57) ]

维度区间    内容                                     维度
─────────────────────────────────────────────────────────
  0 ~ 56    17关键点 (x,y,conf) + 6几何特征         57
 57 ~113    17关键点 (x,y,conf) + 6几何特征 帧间速度  57
114 ~170    17关键点 (x,y,conf) + 6几何特征 帧间加速度 57
```

**设计意图**: 位置描述"在哪里"，速度描述"往哪走"，加速度描述"趋势变化"。跌倒瞬间加速度会出现剧烈变化，是关键的区分信号。6 个几何特征（重心高度、躯干倾斜角、膝角等）的速度和加速度变化也是重要的跌倒前兆。

### 4.6 LSTM 分类器

**推理器**: `classifier/lstm_classifier.py` → `LSTMClassifier` + `LSTMModel`
**模型定义**: `classifier/classifier/risk_lstm_multitask.py` → `RiskLSTM`

**LSTMModel 结构（V3 推理用）**:

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
Take last timestep: (batch, 128)
    │
    ├── risk_head:  Linear(128→1) → Sigmoid → risk ∈ [0,1]
    │
    └── time_head:  Linear(128→1) → ReLU    → time ≥ 0
```

**RiskLSTM 结构（V3.1 训练用）**:

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
    ├── risk_head:  Linear(64→32) → ReLU → Linear(32→1) → logits
    │
    └── time_head:  Linear(64→32) → ReLU → Linear(32→1) → regression
```

**输出**:

| 字段 | 范围 | 含义 |
|------|------|------|
| `risk` | [0, 1] | 跌倒概率，越接近 1 越危险 |
| `time` | [0, ∞) | 距离跌倒完成的帧数，0 = 已跌倒 |
| `label` | {0, 1} | risk > 0.5 → 1（跌倒），否则 0（安全） |

### 4.7 RiskEngine — 风险评估引擎

**文件**: `pipeline/risk_engine.py` | **类**: `RiskEngine`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `smooth_alpha` | 0.6 | EMA 平滑系数（越大响应越快）|
| `warning_thres` | 0.35 | WARNING 状态触发阈值 |
| `danger_thres` | 0.6 | DANGER 状态触发阈值 |
| `trigger_frames` | 5 | 连续危险帧数阈值 |
| `history_size` | 10 | 历史记录长度 |

**三阶段处理**:

```
阶段1 — EMA 平滑:
    smooth[t] = 0.6 × prob + 0.4 × smooth[t-1]

阶段2 — 连续触发计数:
    if smooth >= 0.6: counter += 1
    else:             counter = 0

阶段3 — 状态机判定:
    counter ≥ 5  →  DANGER  确认跌倒
    smooth ≥ 0.35 → WARNING  跌倒预警
    其他          →  SAFE    安全
```

```
SAFE ──(smooth≥0.35)──► WARNING ──(counter≥5)──► DANGER
  ▲                         │                       │
  └───(smooth<0.35)─────────┘                       │
  └─────────────(smooth<0.35)───────────────────────┘
```

---

## 5. 数据集 & 训练

### 5.1 数据源：Le2i 跌倒数据集

| 场景 | 视频数 | 说明 |
|------|--------|------|
| Coffee_room_01 | 48 | 咖啡室视角 1 |
| Coffee_room_02 | 22 | 咖啡室视角 2 |
| Home_01 | 30 | 居家场景 1 |
| Home_02 | 30 | 居家场景 2 |
| **总计** | **130** | |

### 5.2 数据集构建流程

```
Le2i 原始视频 (.avi)
    │
    ▼
YOLO11n-Pose 逐帧提取 → keypoints(T, 17, 2) + confs(T, 17)
    │                    缓存到 yolo_keypoints/ 目录
    ▼
FeatureBuilder 逐帧构建 57 维特征 → pos(T, 57)
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
保存 x.npy, risk.npy, time.npy
```

**关键参数**: `pre_fall = fall_start − 120`（跌倒前 120 帧 ≈ 4.8 秒开始标记风险），`MAX_TIME = 60`（risk 线性衰减区间）。

### 5.3 标签计算规则

| 帧区间 | risk | time | 类别 |
|--------|------|------|------|
| 末帧 < fall_start − 120 | 0 | 999 | normal |
| fall_start−120 ≤ 末帧 < fall_end | 1.0 − time/60 | fall_end−1−末帧 | fall_process |
| 末帧 ≥ fall_end | 1.0 | 0 | fall_after |

**risk 公式**: `risk = max(0.0, 1.0 − time / 60)` — 线性衰减，越接近跌倒越接近 1.0。

### 5.4 训练配置

| 参数 | V3 (train_lstm.py) | V3.1 (train_risk_lstm.py) |
|------|--------------------|-----------------------------|
| Input Dim | 171 | 171 |
| Epochs | 50 | 30 |
| Batch Size | 128 | 32 |
| Learning Rate | 1e-3 | 1e-3 |
| Optimizer | Adam | Adam (weight_decay=1e-5) |
| Scheduler | CosineAnnealingLR | ReduceLROnPlateau |
| Loss (risk) | MSE | BCEWithLogitsLoss |
| Loss (time) | SmoothL1 | SmoothL1 |
| λ_time | 0.1 | 0.5 |
| Gradient Clip | — | max_norm=5.0 |
| 归一化 | Z-score (train set) | Z-score (train set) |

### 5.5 训练结果

| 指标 | 值 |
|------|-----|
| Risk MAE（总体） | 0.0341 |
| Risk MAE（fall） | 0.0637 |
| Risk MAE（normal） | 0.0181 |
| Time MAE（fall） | 2.21 帧 |

### 5.6 数据集统计

| 类别 | 样本数 | 占比 |
|------|--------|------|
| Normal | 4,737 | 65.4% |
| Fall Process | 973 | 13.4% |
| Fall After | 1,533 | 21.2% |
| **总计** | **7,243** | 100% |

---

## 6. 模型权重清单

| 文件 | 大小 | 用途 |
|------|------|------|
| `models/yolo11n-pose.pt` | 6.3 MB | YOLO11n 姿态估计 |
| `models/lstm_multitask.pt` | 1.0 MB | LSTM 多任务（V3，102维→risk+time）|
| `models/lstm_multitask1.pt` | 1.0 MB | LSTM 多任务（V1 备份）|
| `models/lstm_baseline.pt` | 868 KB | LSTM 基线模型 |
| `models/norm_params.npz` | 1.3 KB | Z-score 归一化 mean/std |

---

## 7. 运行方式

### 7.1 环境准备

```bash
conda activate yolo
cd D:\myproject\elder
```

### 7.2 实时检测

```bash
# 检测视频文件
python realtime_detect.py "test/video (1).avi"

# 检测摄像头
python realtime_detect.py
```

### 7.3 诊断模式

```bash
python diagnose.py
# 逐帧打印 risk/time/label，用于调参分析
```

### 7.4 训练模型

```bash
# V3 训练（使用预提取 keypoints 构建数据集 → 训练）
python build_dataset_yolo.py   # 步骤1: 用YOLO提取keypoints + 构建数据集
python train_lstm.py           # 步骤2: 训练LSTM

# V3.1 训练（使用 Le2IDataset）
python scripts/train_risk_lstm.py
```

### 7.5 运行测试

```bash
python scripts/test_pose_extractor.py
python scripts/test_tracker.py
python scripts/test_sequence_buffer.py
python scripts/test_lstm_classifier.py
python scripts/test_fall_detector.py
```

---

## 8. 关键设计决策

### 8.1 为什么用 171 维而非简单坐标？

仅用 17 关键点的 (x,y) 坐标（34 维）无法区分"站立不动"和"正在跌倒但此刻位置恰好正常"。171 维 = 3 × 57（位置+速度+加速度），其中 57 维含 6 个物理意义的几何特征（重心高度、躯干倾斜角、膝角等）。速度捕获运动方向/速率，加速度捕获趋势变化——跌倒瞬间的加速度峰值和几何特征的突变是关键区分信号。

### 8.2 为什么 risk 用 1/(1+time) 而非 0/1 二值？

二值标签导致模型跳变输出，无法渐进式预警。1/(1+time) 提供连续概率：
- 距跌倒 90 帧: risk ≈ 0.011（几乎安全）
- 距跌倒 10 帧: risk ≈ 0.091（高度预警）
- 已跌倒: risk = 1.0（确认）

### 8.3 为什么 pre_fall = fall_start − 90？

跌倒前人体已出现不稳定姿态。90 帧（约 3.6 秒 @25fps）确保模型学习到跌倒前兆信号，同时避免将正常行走误判为危险。

### 8.4 为什么 time 损失只对 fall 样本计算？

Normal 样本 time=999 是占位值，无实际意义。只对 fall 样本计算 time 损失，模型专注于学习"距跌倒还有多远"这个有意义的回归目标。

### 8.5 训练 & 推理必须用同一个 Pose 模型

不同姿态模型的 keypoints 数值分布不同。统一使用 YOLO11n-Pose 确保训练/推理分布一致。

---

## 9. 版本演进

| 版本 | 核心思想 | 输出 |
|------|----------|------|
| V1/V2 | 动作分类 | SAFE / FALL |
| V3 | 风险回归 | risk_score ∈ [0,1] |
| V3.1 | 时间事件预测 | risk + time_to_fall 双头 |

**本质变化**: 从"跌倒后识别"升级为"跌倒前预警"。

---

## 10. 已知限制 & 改进方向

| 问题 | 当前状态 | 改进方向 |
|------|----------|----------|
| Centroid Tracker ID 漂移 | 交叉时 ID 可能互换 | DeepSORT / ByteTrack |
| 单人简化（realtime_detect） | person_id 固定为 0 | 接入 FallDetector 多人管线 |
| 无 API 服务 | 仅本地演示 | FastAPI 封装 REST |
| 配置硬编码 | 参数分散在各模块 | 集中 configs/ |
| 缺少 requirements.txt | 环境复现困难 | 生成依赖清单 |
| LSTM 感受野有限 | 30 帧 ≈ 1.2 秒 | Transformer / TCN |
| V3.1 模型未训练 | 仅完成模型定义 (RiskLSTM) | 用 RiskLSTM + Le2IDataset 训练 |
| configs/ 目录为空 | 仅有设计文档提及 | 实现 model_config.py、thresholds.py |

---

## 11. 文件依赖图

```
realtime_detect.py
├── pose.pose_extractor          → PoseExtractor
├── features.feature_builder     → FeatureBuilder
├── sequence.sequence_buffer     → SequenceBufferV3
└── classifier.lstm_classifier   → LSTMClassifier → LSTMModel (input=171)

pipeline/fall_detector.py
├── pose.pose_extractor          → PoseExtractor
├── tracker.person_tracker       → PersonTracker → Track
├── features.feature_builder     → FeatureBuilder (57维/帧)
├── sequence.sequence_buffer     → SequenceBufferV3
├── classifier.lstm_classifier   → LSTMClassifier (input=171)
└── pipeline.risk_engine         → RiskEngine

train_lstm.py (V3)
├── torch (LSTMModel 内联定义, input=171)
└── sklearn.model_selection

scripts/train_risk_lstm.py (V3.1)
├── classifier.classifier.risk_lstm_multitask → RiskLSTM (input=171)
└── dataset.le2i_dataset → Le2IDataset

build_dataset_yolo.py
├── ultralytics.YOLO
├── features.feature_builder     → FeatureBuilder (57维/帧)
└── 构建 171 维特征 → 保存数据集
```

---

> **一句话总结**: 从摄像头 → 17 个骨架关键点 → 57 维/帧几何增强特征（含重心/倾斜角/膝角） → 30 帧运动序列 (171 维) → LSTM 双头预测 → EMA 平滑状态机，实现 **跌倒前 ~3.6 秒渐进式风险预警**，Risk MAE 低至 0.034。
