# Risk 风险值分析文档

## 概述

`risk` 是一个 **0~1** 的连续值，由 LSTM 多任务模型输出，综合评估人体在 30 帧时间窗口内的跌倒风险。

```text
risk < 0.5  →  SAFE     (绿色)
risk 0.5~0.8 → WARNING  (橙色)
risk ≥ 0.8  →  FALL     (红色)
```

---

## 特征体系 (171 维/帧)

单帧原始特征为 **57 维**，经时序缓存拼接位置/速度/加速度后变为 **171 维**。

### 一、57 维单帧特征 (FeatureBuilder)

```
┌─────────────────────────────────────────────────┐
│  17 关键点 × (x, y, conf)  = 51 维基础特征      │
├─────────────────────────────────────────────────┤
│  cg_y          重心高度 (髋关节中点 y 坐标)      │
│  torso_angle   躯干倾斜角 (肩-髋连线与垂线夹角)  │
│  left_knee_angle   左膝角 (髋-膝-踝)             │
│  right_knee_angle  右膝角                        │
│  foot_distance     双脚距离 (步幅)               │
│  body_height       人体高度 (鼻→踝中点)           │
└─────────────────────────────────────────────────┘
```

#### 1. 重心高度 `cg_y`

```python
hip_center = (left_hip + right_hip) / 2
cg_y = hip_center[1]   # 像素 y 坐标，越大越靠画面下方
```

- 站立时 `cg_y` 较高（数值小）
- 跌倒时 `cg_y` 快速下移（数值变大）
- **速度 `Δcg_y`** 和 **加速度 `Δ²cg_y`** 是跌倒的关键信号

#### 2. 躯干倾斜角 `torso_angle`

```python
shoulder_center = (left_shoulder + right_shoulder) / 2
dx = shoulder_center.x - hip_center.x
dy = shoulder_center.y - hip_center.y
torso_angle = arctan2(dx, -dy)  # 单位：度
```

| 姿态 | torso_angle |
|------|------------|
| 直立 | ≈ 0° |
| 轻微倾斜 | 10° ~ 30° |
| 明显倾斜 | 30° ~ 60° |
| 接近水平/躺倒 | 60° ~ 90° |

- 跌倒时躯干会从接近 0° 迅速增大到 60°+
- **躯干倾斜角速度** 是区分"弯腰"和"跌倒"的关键

#### 3. 膝角 `left_knee_angle` / `right_knee_angle`

```python
angle = arccos(dot(hip→knee, ankle→knee))  # 单位：度
```

| 姿态 | 膝角 |
|------|------|
| 直立 | ≈ 180° |
| 行走屈膝 | 120° ~ 160° |
| 跌倒蜷缩 | 90° ~ 120° |

#### 4. 双脚距离 (步幅) `foot_distance`

```python
foot_distance = ||left_ankle - right_ankle||  # 像素距离
```

| 状态 | foot_distance |
|------|--------------|
| 并脚站立 | 小 (< 30px) |
| 正常行走 | 中 (30~80px) |
| 大步/劈叉/跌倒 | 大 (> 80px) |

- 跌倒时双脚可能突然分开（失去支撑）或交叉缠绕

#### 5. 人体高度 `body_height`

```python
body_height = ||nose - ankle_center||
```

- 站立时最大，跌倒过程中快速缩短（投影变小）

---

### 二、171 维运动特征 (SequenceBuffer)

30 帧 57 维特征拼接为 `(30, 57)`，再计算一阶差分（速度）和二阶差分（加速度），沿特征维拼接：

```text
[position(30,57) | velocity(30,57) | acceleration(30,57)] = (30, 171)
```

| 分量 | 含义 | 物理意义 |
|------|------|---------|
| **position** `p(t)` | 当前帧 57 维特征 | 瞬时姿态 |
| **velocity** `v(t) = p(t) - p(t-1)` | 帧间变化量 | 运动速度 |
| **acceleration** `a(t) = v(t) - v(t-1)` | 速度变化量 | 运动加速度 |

---

## Risk 与物理特征的关系

LSTM 模型从 `(30, 171)` 中学到的跌倒模式，可归纳为以下物理特征组合：

### 1. 躯干倾斜角 (torso_angle) 贡献

```
risk ∝ |torso_angle| × |Δ torso_angle|
```

- 倾斜角越大 + 倾斜速度越快 → risk 越高
- 弯腰（慢速倾斜）vs 跌倒（快速倾斜）：速度维度是关键区分

### 2. 重心加速度 (cg_y acceleration) 贡献

```
risk ∝ Δ²cg_y  (向下加速度)
```

- 重心突然向下的加速度是跌倒的最强信号
- 正常坐下：加速度缓；跌倒：加速度陡

### 3. 步幅 (foot_distance) 贡献

```
risk ∝ Δ foot_distance  (步幅突变)
```

- 行走时步幅周期性变化
- 跌倒时步幅可能出现异常（突然变大/变小/不对称）

### 4. 人体高度 (body_height) 贡献

```
risk ∝ -Δ body_height  (高度快速减小)
```

- 跌倒时鼻→踝投影高度迅速减小（人倒下）

### 5. 膝角变化 贡献

```
risk ∝ 膝角减小速度
```

- 跌倒时往往伴随腿部弯曲（失去支撑）

---

## 模型结构

```
输入 (30, 171)
    ↓
LSTM (2层, hidden=128, dropout=0.2)
    ↓
取最后时刻输出 (1, 128)
    ↓
├─ risk_head:  Linear(128→1) + Sigmoid  →  risk ∈ [0,1]
└─ time_head:  Linear(128→1) + ReLU     →  time (距跌倒完成帧数)
```

- **risk_head** 使用 Sigmoid 将输出压缩到 [0,1]
- **time_head** 使用 ReLU 输出非负值，表示距离跌倒完成的剩余帧数（0 = 已跌倒）

---

## 运动门控 (_check_motion)

推理前有一层运动门控，避免对静止站立的人运行 LSTM：

```
if displacement < 8px AND torso_angle ≤ 30°:
    risk = 0.0   # 静止直立 → 强制 SAFE
else:
    risk = LSTM.predict()  # 有运动或躯干倾斜 → 正常推理
```

其中 `displacement` = 30 帧窗口首尾帧 nose_x 和 hip_y 的欧氏位移。

---

## Risk 值调试

运行时会打印每帧的 risk、time、label：

```text
[F0043] detect=1 keep=1 | buf=30 risk=0.152 time=36.8 label=0
[F0055] detect=1 keep=1 | buf=30 risk=0.559 time=24.9 label=1
[F0078] detect=1 keep=1 | buf=30 risk=0.981 time=0.0  label=1
```

- `time` 值从大变小 → 正在接近跌倒完成点
- `time` = 0 且 risk 高 → 已处于跌倒状态
