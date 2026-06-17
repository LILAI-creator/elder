# 🧠 Elder Fall Risk Prediction System

> 基于 YOLO11n-Pose + LSTM 多任务深度学习的实时老年人跌倒风险预测系统

**版本**: v3.1 | **Risk MAE**: 0.034 | **提前预警**: ~3.6 秒

---

## 快速开始

```bash
conda activate yolo
cd D:\myproject\elder

# 实时检测（视频文件）
python realtime_detect.py "test/video (1).avi"

# 实时检测（摄像头）
python realtime_detect.py
```

---

## 系统流程

```
摄像头 → YOLO11n-Pose (17关键点) → FeatureBuilder (57维/帧)
→ SequenceBuffer (30帧) → 拼接[位置+速度+加速度] (30,171)
→ 2层LSTM → {risk, time} → RiskEngine → SAFE/WARNING/DANGER
```

## 关键指标

| 指标 | 值 |
|------|-----|
| Risk MAE（总体） | 0.0341 |
| Risk MAE（跌倒） | 0.0637 |
| Risk MAE（正常） | 0.0181 |
| Time MAE（跌倒） | 2.21 帧 |
| 数据集规模 | 7,243 样本 (130 视频) |

## 特征维度

每帧 **57 维** = 17×(x,y,conf) + 6 几何特征（重心高度、躯干倾斜角、左/右膝角、双脚距离、人体高度）

时序输入 **(30, 171)** = [pos(30,57) | vel(30,57) | acc(30,57)]

## 模型权重

| 文件 | 大小 | 用途 |
|------|------|------|
| `models/yolo11n-pose.pt` | 6.3 MB | YOLO11n 姿态估计 |
| `models/lstm_multitask.pt` | 1.1 MB | LSTM 多任务（171维→risk+time） |
| `models/norm_params.npz` | 1.9 KB | Z-score 归一化参数 |

## 文档

| 文档 | 说明 |
|------|------|
| [GROUP_MEETING_REPORT.md](GROUP_MEETING_REPORT.md) | 组会汇报文档 |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 项目结构 & 模块详解 |
| [PROJECT_DOC.md](PROJECT_DOC.md) | 技术详细文档 |
| [TDD-Elder-Fall-Detection-202606041000.md](TDD-Elder-Fall-Detection-202606041000.md) | TDD 设计文档 |

## 训练

```bash
# 步骤1: 用YOLO提取keypoints + 构建171维数据集
python build_dataset_yolo.py

# 步骤2: 训练LSTM多任务模型
python train_lstm.py

# V3.1 训练（使用RiskLSTM）
python scripts/train_risk_lstm.py
```

## 测试

```bash
python scripts/test_pose_extractor.py
python scripts/test_tracker.py
python scripts/test_sequence_buffer.py
python scripts/test_lstm_classifier.py
python scripts/test_fall_detector.py
```
