# 老年人跌倒检测系统 — 技术设计文档

## 1. 文档信息

| 项目 | 内容 |
|------|------|
| 项目名称 | Elder Fall Detection System（老年人跌倒检测系统） |
| 文档类型 | TDD（Technical Design Document） |
| 版本 | v1.0 |
| 日期 | 2026-06-04 |
| 状态 | 已实现 |
| 开发环境 | Conda `yolo` (Python 3.9) |
| IDE | VSCode + CodeArts |

---

## 2. 项目概述

### 2.1 背景

老年人跌倒是最常见且最危险的家庭意外之一。本项目构建一套基于**姿态估计 + 时序深度学习**的实时跌倒检测系统，通过摄像头捕获人体姿态，利用 LSTM 对动作序列进行分类，实现从视频流到风险预警的端到端流水线。

### 2.2 目标

- 实时多人姿态检测与跟踪
- 时序动作建模（30帧滑动窗口）
- 二分类推理：Safe / Danger
- 三级风险预警：SAFE / WARNING / DANGER
- 工业级风险评估（EMA 平滑 + 连续触发 + 状态机）

### 2.3 版本状态

| 版本 | 状态 | 说明 |
|------|------|------|
| v1 | **已实现** | YOLO-Pose + Centroid Tracker + 34维特征 + LSTM + Risk Engine |
| v2 | 设计完成 | 64维特征 + 三分类 + GRU/TCN，详见 `xiangmu.md` |
| v3.1 | 设计完成 | 96~128维特征 + 双头模型(Risk+Time-to-Fall)，详见 `v3.1md` |

---

## 3. 技术栈

| 类别 | 技术选型 | 版本/说明 |
|------|---------|----------|
| 开发语言 | Python | 3.9 |
| 深度学习框架 | PyTorch | LSTM 训练与推理 |
| 姿态估计 | Ultralytics YOLO11n-Pose | 17个COCO关键点 |
| 计算机视觉 | OpenCV (cv2) | 视频读取、画面渲染 |
| 数据处理 | NumPy | 特征处理、数据集构建 |
| 机器学习工具 | scikit-learn | train_test_split, classification_report, confusion_matrix |
| 开发环境 | Conda | 环境 `yolo` |
| 依赖管理 | [TBD: 缺失 requirements.txt] | 需补充 |

---

## 4. 系统架构

### 4.1 整体流水线

`Camera Frame → PoseExtractor → PersonTracker → FeatureBuilder → SequenceBuffer → LSTMClassifier → RiskEngine → Alert`

### 4.2 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     FallDetector (pipeline)                      │
│                                                                 │
│  Frame ──► PoseExtractor ──► PersonTracker ──► FeatureBuilder  │
│              (YOLO11n)        (Centroid)        (34-dim)        │
│                                                                 │
│            ──► SequenceBuffer ──► LSTMClassifier ──► RiskEngine │
│               (deque,30)         (2-layer LSTM)    (EMA+SM)    │
│                                                                 │
│            ──► Result: {id, bbox, safe, danger, label,          │
│                        risk_score, state}                       │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 目录结构

| 目录/文件 | 功能说明 |
|-----------|---------|
| `pose/pose_extractor.py` | YOLO-Pose 姿态关键点提取 |
| `tracker/person_tracker.py` | Centroid 多目标跟踪器 |
| `tracker/track.py` | 单目标 Track 数据结构 |
| `features/feature_builder.py` | 34维特征向量构建 |
| `sequence/sequence_buffer.py` | 多人动作序列缓存 (deque maxlen=30) |
| `classifier/lstm_classifier.py` | LSTM 推理器封装 |
| `pipeline/fall_detector.py` | 主流水线（串联所有模块） |
| `pipeline/risk_engine.py` | 风险评估引擎 |
| `models/` | 模型权重 (YOLO, LSTM, norm_params) |
| `dataset/` | 训练数据 (X.npy, Y.npy) |
| `scripts/` | 构建、测试、演示脚本 |
| `train_lstm.py` | LSTM 训练入口 |
| `fall_predict.py` | 早期版核心函数库 |
| `realtime_demo.py` | 早期版实时演示 |

---

## 5. 模块详细设计

### 5.1 PoseExtractor — 姿态提取

**文件**: `pose/pose_extractor.py`

**职责**: 使用 YOLO11n-Pose 从单帧图像中提取所有人员的17个COCO关键点。

**接口**:

- `__init__(model_path="./models/yolo11n-pose.pt")` — 加载模型
- `extract(frame) → List[Dict]` — 输入 OpenCV frame，返回:
  - `bbox`: ndarray(4,) — 边界框 [x1, y1, x2, y2]
  - `keypoints`: ndarray(17,2) — 17个关键点 (x,y) 坐标
  - `score`: float — 检测置信度

**模型**: Ultralytics YOLO11n-Pose，17个COCO关键点（鼻、眼、耳、肩、肘、腕、髋、膝、踝）

### 5.2 PersonTracker — 多目标跟踪

**文件**: `tracker/person_tracker.py`, `tracker/track.py`

**职责**: Centroid Tracking 算法，为画面中的每个人分配固定 ID，管理丢失/新增目标。

**核心参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_missing` | 30 | 最大容忍丢失帧数 |
| `distance_threshold` | 100 | bbox中心点匹配距离阈值(像素) |

**接口**:

- `update(persons) → (tracks, removed_ids)` — 输入 PoseExtractor 结果，返回跟踪列表和移除ID列表
- 匹配算法：基于 bbox 中心点欧氏距离，匈牙利最优匹配

**Track 数据结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 唯一跟踪 ID |
| `bbox` | ndarray | 边界框 |
| `keypoints` | ndarray | 关键点 |
| `missing` | int | 连续丢失帧数 |
| `age` | int | 存活帧数 |
| `hit_count` | int | 命中帧数 |

**已知局限**: 两人中心点交叉时可能 ID 漂移，v2 计划引入 DeepSORT。

### 5.3 FeatureBuilder — 特征构建

**文件**: `features/feature_builder.py`

**职责**: 将 (17,2) 关键点 + bbox 构建为 34 维特征向量。

**特征构成**:

| 维度范围 | 内容 | 维数 |
|---------|------|------|
| 0-27 | 前14个关键点的 (x,y) 坐标展平 | 28 |
| 28-29 | 人体中心点 (center_x, center_y) | 2 |
| 30-31 | bbox 尺寸 (width, height) | 2 |
| 32 | 宽高比 (aspect_ratio) | 1 |
| 33 | 躯干倾斜角 (torso_angle = arctan2) | 1 |
| **合计** | | **34** |

**接口**: `build(keypoints, bbox) → ndarray(34,)`

### 5.4 SequenceBuffer — 序列缓冲

**文件**: `sequence/sequence_buffer.py`

**职责**: 为每个 person_id 维护固定长度的动作序列缓存。

**核心参数**: `maxlen=30`（deque 最大长度，即30帧时序窗口）

**接口**:

| 方法 | 说明 |
|------|------|
| `update(person_id, feature)` | 添加一帧34维特征到对应person_id的缓存 |
| `is_ready(person_id) → bool` | 判断是否已累积30帧 |
| `get_sequence(person_id) → ndarray(30,34)` | 获取完整序列 |
| `remove(person_id)` | 删除指定人员缓存（与 Tracker 同步） |
| `clear()` | 清空所有缓存 |

### 5.5 LSTMClassifier — LSTM 分类器

**文件**: `classifier/lstm_classifier.py`

**职责**: 加载训练好的 LSTM 模型进行推理，输出 Safe/Danger 概率。

**模型结构**:

| 层 | 参数 |
|----|------|
| LSTM Layer 1 | input_size=34, hidden_size=128, dropout=0.2 |
| LSTM Layer 2 | hidden_size=128, dropout=0.2 |
| Fully Connected | in=128, out=2 |
| Softmax | 二分类概率 |

**接口**:

| 方法 | 说明 |
|------|------|
| `predict(sequence) → Dict` | 输入 (30,34) → 返回 `{safe, danger, label}` |
| `predict_proba(sequence) → float` | 仅返回 danger 概率 |
| `predict_label(sequence) → int` | 仅返回标签 (0=Safe, 1=Danger) |

**依赖文件**:
- `models/lstm_baseline.pt` — 模型权重
- `models/norm_params.npz` — Z-score 归一化参数 (mean, std)

### 5.6 RiskEngine — 风险评估引擎

**文件**: `pipeline/risk_engine.py`

**职责**: 工业级风险评估，输出 SAFE / WARNING / DANGER 三态。

**核心机制**:

1. **EMA 平滑**: `smooth = alpha * danger_prob + (1 - alpha) * prev`，alpha=0.6
2. **连续触发计数**: 当 smooth >= danger_thres(0.6) 时计数器+1，否则归零
3. **三态状态机**:
   - counter >= trigger_frames(5) → **DANGER**
   - smooth >= warning_thres(0.35) → **WARNING**
   - 其他 → **SAFE**

**接口**:

- `update(danger_prob) → Dict` — 返回 `{state, smooth_score, counter}`

### 5.7 FallDetector — 主流水线

**文件**: `pipeline/fall_detector.py`

**职责**: 串联所有模块，提供单帧处理入口。

**`process(frame)` 流程**:

1. PoseExtractor 提取所有人员关键点
2. PersonTracker 跟踪并分配 ID
3. 清理已消失 ID 的 SequenceBuffer 和 RiskEngine 缓存
4. 对每个 Track:
   - FeatureBuilder 构建 34 维特征
   - SequenceBuffer 累积特征
   - 若 is_ready: LSTMClassifier 推理
   - RiskEngine 评估风险状态
5. 返回每个人的检测结果列表

**返回结构**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 人员 ID |
| `bbox` | ndarray | 边界框 |
| `safe` | float | Safe 概率 |
| `danger` | float | Danger 概率 |
| `label` | int | 分类标签 |
| `risk_score` | float | EMA 平滑后的风险分数 |
| `state` | str | 风险状态 (SAFE/WARNING/DANGER) |

---

## 6. 模型训练

### 6.1 训练脚本

**文件**: `train_lstm.py`

**训练参数**:

| 参数 | 值 |
|------|-----|
| Device | CUDA 优先，fallback CPU |
| Epochs | 20 |
| Batch Size | 128 |
| Learning Rate | 1e-3 |
| LR Scheduler | CosineAnnealing |
| Loss | 加权 CrossEntropyLoss（处理样本不平衡） |
| Train/Test Split | 80/20 (stratified) |

**模型输出**:

| 文件 | 说明 |
|------|------|
| `models/lstm_baseline.pt` | LSTM 模型权重 (state_dict) |
| `models/norm_params.npz` | Z-score 归一化参数 (训练集 mean/std) |

### 6.2 数据集构建

**脚本**: `scripts/build_le2i_dataset.py`

**数据源**: Le2i 跌倒检测数据集（`D:/my_datasets/Le2i/`）

**关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| WINDOW | 30 | 滑动窗口长度 |
| PREFALL | 25 | 跌倒前25帧标记为 Danger |

**流程**:

1. 遍历 `fall/` 目录的预提取关键点文件
2. 读取对应标注文件获取 `fall_start` / `fall_end` 帧号
3. 对跌倒视频：从 `fall_start - PREFALL` 开始标记为 Danger(1)
4. 对正常视频：全部标记为 Safe(0)
5. 滑动窗口切分：每30帧为一个样本，标签取窗口最后一帧
6. 保存为 `dataset/X.npy` (shape: N×30×34) 和 `dataset/Y.npy`

---

## 7. 模型文件清单

| 文件路径 | 类型 | 说明 |
|---------|------|------|
| `models/yolo11n-pose.pt` | YOLO11n-Pose | Ultralytics YOLO11 nano 姿态估计模型，17个COCO关键点 |
| `models/lstm_baseline.pt` | PyTorch state_dict | 2层LSTM分类器权重 (input=34, hidden=128, output=2) |
| `models/norm_params.npz` | NumPy 压缩文件 | Z-score 归一化参数 (mean, std)，推理时特征归一化使用 |

---

## 8. 数据集

### 8.1 项目内训练数据

| 文件 | Shape | 说明 |
|------|-------|------|
| `dataset/X.npy` | (N, 30, 34) | N个30帧滑动窗口，每帧34维特征 |
| `dataset/Y.npy` | (N,) | 标签，0=Safe, 1=Danger |

### 8.2 外部数据集

| 数据集 | 路径 | 说明 |
|--------|------|------|
| Le2i | `D:/my_datasets/Le2i/Le2i/Le2i/` | 多场景跌倒视频 + 标注文件 |
| Le2i 关键点 | `D:/my_datasets/le2i_keypoints/le2i_keypoints/` | 预提取关键点 (fall/ + normal/) |
| UR-FALL | `D:/my_datasets/UR-FALL/` | 另一跌倒检测数据集（部分引用） |

### 8.3 测试视频

| 文件 | 说明 |
|------|------|
| `test/video (1).avi` | Le2i 测试视频 (~35MB) |
| `test/video (22).avi` | Le2i 测试视频 (~52MB) |

---

## 9. 使用方式

### 9.1 环境准备

```bash
conda activate yolo
# [TBD: 需补充 requirements.txt]
# 所需依赖: ultralytics, torch, numpy, opencv-python, scikit-learn
```

### 9.2 数据集构建

```bash
python scripts/build_le2i_dataset.py
```

### 9.3 模型训练

```bash
python train_lstm.py
```

### 9.4 实时推理（推荐 — 流水线版）

```bash
python scripts/realtime_demo.py   # 默认使用 test/video (1).avi
```

### 9.5 实时推理（早期版）

```bash
python realtime_demo.py --video "path/to/video.avi"   # 视频模式
python realtime_demo.py                                # 摄像头模式
```

### 9.6 模块测试脚本

| 脚本 | 用途 |
|------|------|
| `scripts/test_pose_extractor.py` | 摄像头实时测试 PoseExtractor，绘制 bbox 和关键点 |
| `scripts/test_tracker.py` | 摄像头实时测试 PersonTracker，显示 ID 和 bbox |
| `scripts/test_lstm_classifier.py` | 用随机数据验证 LSTMClassifier 加载和推理 |
| `scripts/test_sequence_buffer.py` | SequenceBuffer 单元测试 |
| `scripts/test_fall_detector.py` | FallDetector 全流程测试 |
| `test_pose.py` | 最简化 YOLO-Pose 摄像头测试 |

---

## 10. 非功能设计

### 10.1 性能

- YOLO11n-Pose：nano 版本，推理速度较快（具体 FPS 取决于 GPU）
- LSTM 推理：极轻量（2层LSTM, hidden=128），单次推理 <1ms
- 视频演示以 0.25 倍速播放，便于观察

### 10.2 可扩展性

- 模块化架构，各模块独立可替换
- v2 规划：64维特征 + 三分类 + GRU/TCN
- v3.1 规划：96~128维特征 + 双头模型(Risk + Time-to-Fall) + Transformer

### 10.3 已知问题

| 问题 | 位置 | 影响 | 建议 |
|------|------|------|------|
| Centroid Tracker ID 漂移 | `tracker/person_tracker.py` | 两人交叉时 ID 可能互换 | v2 引入 DeepSORT |
| `fall_predict.py` 第42行代码误粘贴 | `fall_predict.py:42` | 早期版推理可能报错 | 使用 `scripts/realtime_demo.py` |
| 缺少 `requirements.txt` | 项目根目录 | 环境复现困难 | 需补充依赖文件 |
| `configs/config.py` 为空 | `configs/config.py` | 参数硬编码 | 集中配置管理 |
| 无 API 服务 | 项目 | 无法远程调用 | 需开发 Flask/FastAPI 服务 |

---

## 11. 部署架构

当前为本地开发部署，无服务化部署。

**v2/v3.1 规划**:

- ONNX 导出 + TensorRT 加速
- 多线程 Pipeline（生产者-消费者模式）
- Flask/FastAPI API 服务
- 边缘设备部署（Jetson 等）

---

## 12. 风险与待决项

| 项目 | 说明 | 优先级 |
|------|------|--------|
| 补充 `requirements.txt` | 环境依赖未管理 | 高 |
| 修复 `fall_predict.py` 误粘贴 | 早期版代码错误 | 中 |
| Centroid Tracker 局限 | ID 漂移问题 | 中（v2 解决） |
| 配置集中化 | `configs/config.py` 空文件，参数分散 | 低 |
| API 服务开发 | 无远程调用能力 | 低（按需） |

---

## 13. 附录

### 13.1 COCO 17关键点定义

| 序号 | 关键点 | 序号 | 关键点 |
|------|--------|------|--------|
| 0 | 鼻 (nose) | 9 | 左膝 (left_knee) |
| 1 | 左眼 (left_eye) | 10 | 左踝 (left_ankle) |
| 2 | 右眼 (right_eye) | 11 | 右髋 (right_hip) |
| 3 | 左耳 (left_ear) | 12 | 右膝 (right_knee) |
| 4 | 右耳 (right_ear) | 13 | 右踝 (right_ankle) |
| 5 | 左肩 (left_shoulder) | 14 | 左髋 (left_hip) * |
| 6 | 右肩 (right_shoulder) | 15 | 右髋 (right_hip) * |
| 7 | 左肘 (left_elbow) | 16 | 左髋 (left_hip) * |
| 8 | 右肘 (right_elbow) | | |

*注: 序号14-16为COCO标准关键点，特征构建时使用前14个关键点。

### 13.2 项目设计文档索引

| 文件 | 版本 | 说明 |
|------|------|------|
| `xiangmu.md` | v2 | 64维特征 + 三分类 + 工业级 Risk Engine |
| `v3.1md` | v3.1 | 双头模型(Risk+Time-to-Fall) + 96~128维特征 |
| `question.md` | - | 当前问题与改进计划 |
| `v3.md` | v3 | 占位空文件 |
