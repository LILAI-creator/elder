# 老年人跌倒检测系统 - 技术文档

## 1. 项目概述

本系统是一套基于**姿态估计 + 时序深度学习**的实时老年人跌倒检测系统。通过摄像头捕获人体姿态关键点，利用LSTM多任务模型对动作序列进行分析，同时预测**跌倒风险概率(risk)**和**距离跌倒完成的时间(time)**，实现从视频流到风险预警的端到端流水线。

### 1.1 核心能力

| 能力 | 说明 |
|------|------|
| 实时姿态提取 | YOLO11n-Pose，17个COCO关键点，单帧推理 |
| 多目标跟踪 | Centroid Tracking，为每人分配固定ID |
| 时序动作分析 | 2层LSTM，30帧滑动窗口，102维特征输入 |
| 多任务预测 | 同时输出risk(跌倒概率)和time(距跌倒时间) |
| 风险评估 | EMA平滑 + 连续触发 + 三态状态机(SAFE/WARNING/DANGER) |

### 1.2 技术栈

| 组件 | 技术 |
|------|------|
| 姿态估计 | Ultralytics YOLO11n-Pose |
| 时序模型 | PyTorch LSTM (2层, hidden=128) |
| 多目标跟踪 | Centroid Tracking |
| 图像处理 | OpenCV |
| 数据处理 | NumPy, scikit-learn |
| 开发环境 | Conda (yolo) |

---

## 2. 系统架构

### 2.1 整体流水线

```
Camera Frame (BGR)
    │
    ▼
┌─────────────────────────────────┐
│  1. PoseExtractor               │  YOLO11n-Pose 提取17个COCO关键点
│     pose/pose_extractor.py      │  输入: frame (H,W,3)
│                                 │  输出: [{bbox(4,), keypoints(17,2), score}]
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  2. PersonTracker               │  Centroid Tracking 多目标跟踪
│     tracker/person_tracker.py   │  输入: [{bbox, keypoints, score}]
│                                 │  输出: (tracks, removed_ids)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  3. FeatureBuilder              │  关键点展平为34维特征
│     pose/pose_extractor.py      │  输入: keypoints(17,2)
│     build_feature()             │  输出: feature(34,)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  4. SequenceBuffer              │  30帧deque缓存 + 速度/加速度计算
│     sequence/sequence_buffer.py │  输入: (person_id, feature(34,))
│                                 │  输出: raw(30,34), vel(30,34), acc(30,34)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  5. Feature Concatenation       │  拼接 [位置, 速度, 加速度]
│     realtime_detect.py          │  输入: raw(30,34), vel(30,34), acc(30,34)
│     build_102_feature()         │  输出: sequence(30,102)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  6. LSTMClassifier              │  多任务LSTM推理
│     classifier/lstm_classifier.py│  输入: sequence(30,102)
│                                 │  输出: {risk: float, time: float, label: int}
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  7. RiskEngine                  │  EMA平滑 + 连续触发 + 三态状态机
│     pipeline/risk_engine.py     │  输入: (person_id, danger_prob)
│                                 │  输出: {risk_score: float, state: SAFE/WARNING/DANGER}
└─────────────────────────────────┘
```

### 2.2 项目目录结构

```
D:\myproject\elder\
│
├── pose/                          # 姿态提取模块
│   ├── __init__.py
│   └── pose_extractor.py          # YOLO11n-Pose关键点提取
│
├── tracker/                       # 多目标跟踪模块
│   ├── __init__.py
│   ├── track.py                   # 单目标Track数据结构
│   └── person_tracker.py          # Centroid多目标跟踪器
│
├── features/                      # 特征工程模块
│   ├── __init__.py
│   ├── feature_builder.py         # V3特征构建器(51维)
│   └── label_generator_v3.py      # V3标签生成器
│
├── sequence/                      # 序列缓冲模块
│   ├── __init__.py
│   └── sequence_buffer.py         # 30帧deque缓存器
│
├── classifier/                    # 分类器模块
│   ├── lstm_classifier.py         # LSTM推理器(102维输入, risk+time双头)
│   └── classifier/
│       └── risk_lstm_multitask.py # V3.1多任务LSTM模型定义
│
├── pipeline/                      # 流水线模块
│   ├── __init__.py
│   ├── fall_detector.py           # 主流水线(串联所有模块)
│   └── risk_engine.py             # 风险评估引擎(EMA+状态机)
│
├── models/                        # 模型权重
│   ├── yolo11n-pose.pt            # YOLO11n-Pose姿态估计模型
│   ├── lstm_multitask.pt          # LSTM多任务模型权重
│   └── norm_params.npz            # Z-score归一化参数(mean, std)
│
├── build_dataset.py               # 数据集构建(从预提取keypoints)
├── build_dataset_yolo.py          # 数据集构建(用YOLO重新提取keypoints)
├── train_lstm.py                  # LSTM多任务模型训练
├── realtime_detect.py             # 实时检测演示
├── diagnose.py                    # 诊断脚本
└── scripts/                       # 辅助脚本
    ├── realtime_demo.py           # 流水线版实时演示
    ├── test_pose_extractor.py     # PoseExtractor测试
    ├── test_tracker.py            # PersonTracker测试
    └── ...
```

---

## 3. 核心模块详解

### 3.1 PoseExtractor - 姿态提取

**文件**: `pose/pose_extractor.py`

**模型**: YOLO11n-Pose (Ultralytics)

**功能**: 从单帧图像中提取所有人体实例的17个COCO关键点

**COCO 17关键点定义**:

| 编号 | 部位 | 编号 | 部位 | 编号 | 部位 |
|------|------|------|------|------|------|
| 0 | 鼻子 | 6 | 左髋 | 12 | 左髋 |
| 1 | 左眼 | 7 | 左膝 | 13 | 左膝 |
| 2 | 右眼 | 8 | 左踝 | 14 | 左踝 |
| 3 | 左耳 | 9 | 右髋 | 15 | 左脚跟 |
| 4 | 右耳 | 10 | 右膝 | 16 | 右脚跟 |
| 5 | 左肩 | 11 | 右踝 | | |

**输入/输出**:

```
输入:  frame (ndarray, H×W×3, BGR)
输出:  [
    {
        "bbox": ndarray(4,),       # [x1, y1, x2, y2]
        "keypoints": ndarray(17,2), # (x, y) 坐标
        "score": float             # 检测置信度
    },
    ...
]
```

**特征构建** (`build_feature`方法):

```
keypoints(17,2) → reshape → feature(34,)
```

### 3.2 PersonTracker - 多目标跟踪

**文件**: `tracker/person_tracker.py`, `tracker/track.py`

**算法**: Centroid Tracking (质心匹配)

**核心逻辑**:

1. 对当前帧每个检测到的人，计算其bbox中心点
2. 与已有Track的bbox中心点计算欧氏距离
3. 距离小于阈值(默认100像素)则匹配，更新Track
4. 未匹配的检测创建新Track，分配新ID
5. 连续丢失超过max_missing帧(默认30)的Track被删除

**Track数据结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| id | int | 唯一标识 |
| bbox | ndarray(4,) | 边界框 [x1,y1,x2,y2] |
| keypoints | ndarray(17,2) | 最新关键点 |
| missing | int | 连续丢失帧数 |
| age | int | 存活总帧数 |
| hit_count | int | 成功匹配次数 |

### 3.3 SequenceBuffer - 序列缓冲

**文件**: `sequence/sequence_buffer.py`

**功能**: 为每个跟踪目标维护一个30帧的deque缓冲区，支持速度和加速度计算

**核心方法**:

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `update(person_id, feature)` | (34,) | - | 追加一帧特征到缓冲区 |
| `is_ready(person_id)` | - | bool | 缓冲区是否已满30帧 |
| `get_sequence(person_id)` | - | (30,34) | 获取原始位置序列 |
| `get_velocity(person_id)` | - | (30,34) | 计算帧间速度: v[t] = pos[t] - pos[t-1] |
| `get_acceleration(person_id)` | - | (30,34) | 计算帧间加速度: a[t] = v[t] - v[t-1] |

**速度/加速度计算**:

```
位置:  pos[t] = keypoints[t]                    (17×2 = 34维)
速度:  vel[t] = pos[t] - pos[t-1]  (t≥1)       (34维)
       vel[0] = 0
加速度: acc[t] = vel[t] - vel[t-1]  (t≥1)       (34维)
       acc[0] = 0
```

**异常过滤**: 自动过滤NaN值和shape不匹配的特征

### 3.4 特征拼接 - 102维特征构建

**位置**: `realtime_detect.py` 中的 `build_102_feature()`

**拼接方式**:

```
sequence(30,102) = [pos(30,34) | vel(30,34) | acc(30,34)]
```

| 维度区间 | 内容 | 维度 |
|----------|------|------|
| 0~33 | 17个关键点的(x,y)坐标 | 34 |
| 34~67 | 17个关键点的(x,y)速度 | 34 |
| 68~101 | 17个关键点的(x,y)加速度 | 34 |

**设计意图**: 位置描述"在哪里"，速度描述"往哪走"，加速度描述"变化趋势"。跌倒时加速度会出现剧烈变化，是关键区分信号。

### 3.5 LSTMClassifier - 多任务LSTM模型

**文件**: `classifier/lstm_classifier.py`

**模型结构**:

```
Input: (batch, 30, 102)
    │
    ▼
LSTM Layer 1 (input=102, hidden=128, dropout=0.2)
    │
    ▼
LSTM Layer 2 (input=128, hidden=128, dropout=0.2)
    │
    ▼
Take last timestep output: (batch, 128)
    │
    ├──► risk_head: Linear(128→1) → Sigmoid → risk ∈ [0, 1]
    │
    └──► time_head: Linear(128→1) → ReLU → time ≥ 0
```

**输出含义**:

| 输出 | 范围 | 含义 |
|------|------|------|
| risk | [0, 1] | 跌倒概率，越接近1表示越可能跌倒 |
| time | [0, +∞) | 距离跌倒完成的帧数，0表示已跌倒 |
| label | {0, 1} | risk > 0.5 时为1(跌倒)，否则为0(安全) |

**归一化**: 推理前对输入进行Z-score归一化: `x_norm = (x - mean) / std`，参数从训练集计算并保存到 `norm_params.npz`

### 3.6 RiskEngine - 风险评估引擎

**文件**: `pipeline/risk_engine.py`

**功能**: 对LSTM输出的risk值进行平滑和状态判定，避免抖动

**三阶段处理**:

#### 阶段1: EMA指数移动平均平滑

```
smooth[t] = α × danger_prob + (1 - α) × smooth[t-1]
```

- α = 0.6 (默认)，越大越响应快，越小越平滑

#### 阶段2: 连续危险帧计数

```
if smooth >= danger_threshold (0.6):
    counter += 1
else:
    counter = 0
```

#### 阶段3: 三态状态机判定

| 条件 | 状态 | 含义 |
|------|------|------|
| counter ≥ trigger_frames (5) | DANGER | 确认跌倒 |
| smooth ≥ warning_threshold (0.35) | WARNING | 跌倒预警 |
| 其他 | SAFE | 安全 |

**状态转换图**:

```
SAFE ──(smooth≥0.35)──► WARNING ──(counter≥5)──► DANGER
  ▲                          │                        │
  └────(smooth<0.35)─────────┘                        │
  └────────────(smooth<0.35)─────────────────────────┘
```

---

## 4. 数据集构建

### 4.1 数据源: Le2i数据集

**路径**: `D:\my_datasets\Le2i\`

**内容**: 多场景监控视频 + 跌倒标注

| 场景 | 视频数 | 有标注 | 说明 |
|------|--------|--------|------|
| Coffee_room_01 | 48 | 48 | 咖啡室1 |
| Coffee_room_02 | 22 | 22 | 咖啡室2 |
| Home_01 | 30 | 30 | 家庭1 |
| Home_02 | 30 | 30 | 家庭2 |

**标注格式** (每个视频对应一个txt文件):

```
第1行: fall_start (跌倒开始帧号)
第2行: fall_end   (跌倒结束帧号)
第3行起: frame_id, status, bbox_x, bbox_y, bbox_w, bbox_h
```

- status=1: 正常行走
- status=7/8: 跌倒后躺在地上
- 若第1/2行无法解析为整数，则为normal视频（无跌倒）

### 4.2 构建流程

**脚本**: `build_dataset_yolo.py`

```
Le2i原始视频(.avi)
    │
    ▼
YOLO11n-Pose 逐帧提取 → keypoints(T, 17, 2) + confs(T, 17)
    │                              ↓ 缓存到 yolo_keypoints/ 目录
    ▼
计算速度/加速度 → 拼接 [pos, vel, acc] → features(T, 102)
    │
    ▼
滑动窗口切分 (window=30, stride=5)
    │
    ▼
计算 risk 和 time 标签
    │
    ▼
保存 x.npy(7243,30,102), risk.npy(7243,1), time.npy(7243,1)
```

### 4.3 标签计算规则

**关键参数**: `pre_fall = fall_start - 90`（跌倒开始前90帧起算"快要跌倒"）

| 帧区间 | risk | time | 类别 |
|--------|------|------|------|
| 末帧 < fall_start - 90 | 0 | 999 | normal (正常) |
| fall_start - 90 ≤ 末帧 < fall_end | 1/(1+time) | fall_end - 1 - 末帧 | fall_process (跌倒过程) |
| 末帧 ≥ fall_end | 1 | 0 | fall_after (跌倒后) |

**risk公式**: `risk = 1 / (1 + time)`

- time=0 → risk=1.0 (已跌倒)
- time=10 → risk=0.091 (10帧后跌倒)
- time=90 → risk=0.011 (90帧后跌倒)

**设计意图**: risk是一个连续概率值，越接近跌倒完成时刻越趋近1，实现渐进式预警而非二值跳变。

---

## 5. 模型训练

### 5.1 训练脚本

**文件**: `train_lstm.py`

**数据集**: `D:\my_datasets\Le2i\processed\` (x.npy, risk.npy, time.npy)

### 5.2 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| EPOCHS | 50 | 训练轮数 |
| BATCH_SIZE | 128 | 批大小 |
| LR | 1e-3 | 初始学习率 |
| WINDOW_SIZE | 30 | 输入序列长度 |
| FEATURE_DIM | 102 | 输入特征维度 |
| HIDDEN_SIZE | 128 | LSTM隐藏层大小 |
| NUM_LAYERS | 2 | LSTM层数 |
| DROPOUT | 0.2 | LSTM dropout |
| λ_time | 0.1 | time损失权重 |

### 5.3 损失函数

**多任务损失**:

```
Loss = Loss_risk + λ_time × Loss_time
     = MSE(pred_risk, true_risk) + 0.1 × SmoothL1(pred_time, true_time)
```

**关键设计**: time损失**只对fall样本计算**（risk > 0的样本），normal样本的time=999是占位值，不参与time损失计算。

| 损失项 | 公式 | 适用范围 |
|--------|------|----------|
| Loss_risk | MSE = mean((pred - true)²) | 所有样本 |
| Loss_time | SmoothL1 = 见下方 | 仅fall样本(risk>0) |

**SmoothL1 Loss**:

```
if |x| < 1:  0.5 × x²
else:        |x| - 0.5
```

比MSE对异常值更鲁棒，适合time这种范围差异大的回归目标。

### 5.4 优化策略

| 策略 | 说明 |
|------|------|
| Z-score归一化 | 对训练集计算mean/std，推理时使用相同参数 |
| CosineAnnealingLR | 余弦退火学习率调度 |
| Best Model保存 | 保存测试集loss最低的模型 |
| train_test_split | 80%训练 / 20%测试，random_state=42 |

### 5.5 模型输出

| 文件 | 内容 |
|------|------|
| `models/lstm_multitask.pt` | PyTorch模型权重(state_dict) |
| `models/norm_params.npz` | 归一化参数(mean, std) |

---

## 6. 实时检测

### 6.1 使用方式

```bash
conda activate yolo

# 检测视频文件
python realtime_detect.py "path/to/video.avi"

# 检测摄像头
python realtime_detect.py
```

### 6.2 检测流程

```
1. 读取一帧 → PoseExtractor提取关键点
2. keypoints(17,2) → reshape → feature(34,)
3. SequenceBuffer.update(person_id, feature) → 缓存30帧
4. 缓冲区满30帧后:
   a. 获取 raw(30,34), vel(30,34), acc(30,34)
   b. 拼接为 sequence(30,102)
   c. Z-score归一化
   d. LSTM推理 → {risk, time, label}
5. 可视化: 绘制bbox + 状态 + risk值 + time值
```

### 6.3 可视化颜色

| 状态 | 颜色 | 条件 |
|------|------|------|
| SAFE | 绿色 | risk ≤ 0.3 |
| WARNING | 橙色 | 0.3 < risk ≤ 0.5 |
| FALL | 红色 | risk > 0.5 |

### 6.4 播放速度

按视频原始FPS控制播放速度，非实时推理时不会过快播放。

---

## 7. 关键设计决策

### 7.1 为什么用102维特征而非34维？

34维只包含位置信息，无法区分"站着不动"和"正在跌倒但恰好此刻位置正常"。加入速度和加速度后：

- **速度**: 捕获运动方向和速率，跌倒时y方向速度会突然增大
- **加速度**: 捕获运动趋势变化，跌倒瞬间加速度会出现尖峰

### 7.2 为什么risk用1/(1+time)而非0/1二值？

二值标签会导致模型输出跳变，无法实现渐进式预警。1/(1+time)提供连续概率：

- 离跌倒90帧: risk ≈ 0.011 (几乎安全)
- 离跌倒30帧: risk ≈ 0.032 (开始预警)
- 离跌倒10帧: risk ≈ 0.091 (高度预警)
- 已跌倒: risk = 1.0 (确认跌倒)

### 7.3 为什么pre_fall = fall_start - 90？

跌倒是一个渐进过程，在fall_start之前人体已经出现不稳定姿态。90帧(约3.6秒@25fps)的提前量确保：

1. 模型能学习到跌倒前的预兆信号
2. 实际应用中有足够的预警时间
3. 避免将正常行走误判为跌倒前兆

### 7.4 为什么time损失只对fall样本计算？

normal样本的time=999是占位值，没有实际意义。如果参与训练会严重干扰模型对time的学习。只对fall样本计算time损失，模型可以专注于学习"距离跌倒还有多远"这个有意义的回归目标。

### 7.5 为什么训练和推理必须用同一个Pose模型？

不同姿态估计模型提取的keypoints在数值范围、精度、偏差上存在差异。用模型A提取训练数据、用模型B推理，会导致分布不一致，模型无法泛化。本项目统一使用YOLO11n-Pose。

---

## 8. 数据流示例

以一个157帧的fall视频为例 (fall_start=48, fall_end=80):

```
帧 0~17:    keypoints全为0 (人未进入画面)
帧 18~47:   正常行走 (末帧 < fall_start-90=0, 但pre_fall=max(0,-42)=0)
            → risk=0, time=999
帧 48~79:   跌倒过程
            → risk=1/(1+(79-末帧)), time=79-末帧
帧 80~156:  跌倒后躺在地上
            → risk=1.0, time=0
```

滑动窗口(30帧, 步长5)切分后，该视频生成约25个样本。

---

## 9. 性能指标

### 9.1 训练结果 (最新)

| 指标 | 值 |
|------|-----|
| Risk MAE (总体) | 0.0341 |
| Risk MAE (fall) | 0.0637 |
| Risk MAE (normal) | 0.0181 |
| Time MAE (fall) | 2.21 帧 |

### 9.2 数据集统计

| 类别 | 样本数 | 占比 |
|------|--------|------|
| Normal | 4737 | 65.4% |
| Fall Process | 973 | 13.4% |
| Fall After | 1533 | 21.2% |
| **总计** | **7243** | 100% |

---

## 10. 已知限制与改进方向

| 问题 | 当前状态 | 改进方向 |
|------|----------|----------|
| Centroid Tracker ID漂移 | 两人交叉时ID可能互换 | 引入DeepSORT/ByteTrack |
| 单人假设 | realtime_detect.py固定person_id=0 | 接入PersonTracker实现多人检测 |
| 无API服务 | 仅支持本地演示 | FastAPI/Uvicorn封装为REST服务 |
| 配置硬编码 | 参数分散在各模块 | 集中到configs/config.py |
| 缺少requirements.txt | 环境复现困难 | 生成依赖清单 |
| LSTM感受野有限 | 30帧≈1.2秒 | 尝试Transformer/TCN扩大感受野 |