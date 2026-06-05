import torch
import torch.nn as nn


class RiskLSTM(nn.Module):
    """
    工业级跌倒检测 LSTM 多任务模型（稳定版）

    设计目标：
    1. 输入30帧人体序列特征 (102维/帧)
    2. 输出：
        - risk：跌倒风险（分类 logits）
        - time：距离跌倒时间（回归）
    """

    def __init__(self,
                 input_dim=102,      # 每帧特征维度（x,y,v,a等拼接）
                 hidden_dim=128,     # LSTM隐藏状态维度（时序表达能力）
                 num_layers=2,       # LSTM堆叠层数（增强时序建模能力）
                 dropout=0.2):       # 防止过拟合

        super(RiskLSTM, self).__init__()

        # ==================================================
        # 1️⃣ LSTM 时序编码器（核心模块）
        # ==================================================
        self.lstm = nn.LSTM(
            input_size=input_dim,       # 输入特征维度 = 102
            hidden_size=hidden_dim,     # 输出隐藏状态维度 = 128
            num_layers=num_layers,      # 2层LSTM叠加（增强表达能力）
            batch_first=True,           # 输入格式：(B, T, F)
            dropout=dropout if num_layers > 1 else 0
                                        # 多层之间使用dropout防过拟合
        )

        # ==================================================
        # 2️⃣ 时序特征压缩层（特征降维 + 稳定训练）
        # ==================================================
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),  # 将128维时序特征压缩到64维
            nn.ReLU(),                  # 非线性增强表达能力

            nn.LayerNorm(64),          # ⭐关键：归一化，稳定梯度分布
                                        # 防止LSTM输出分布漂移

            nn.Dropout(0.2)            # 防止过拟合，提高泛化能力
        )

        # ==================================================
        # 3️⃣ 风险预测头（分类任务）
        # ==================================================
        self.risk_head = nn.Sequential(
            nn.Linear(64, 32),         # 进一步提取任务特征
            nn.ReLU(),                # 非线性变换
            nn.Linear(32, 1)          # 输出1维 logits（⚠️不做sigmoid）
        )
        # ⚠️ 注意：
        # 这里输出的是 logits
        # sigmoid 应该交给 loss (BCEWithLogitsLoss)

        # ==================================================
        # 4️⃣ 时间预测头（回归任务）
        # ==================================================
        self.time_head = nn.Sequential(
            nn.Linear(64, 32),         # 压缩特征
            nn.ReLU(),                # 非线性
            nn.Linear(32, 1)          # 输出连续值（时间）
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: (B, 30, 102)
               B = batch size
               30 = 时间窗口帧数
               102 = 每帧特征维度

        返回:
            risk: (B, 1) logits（未sigmoid）
            time: (B, 1) 回归值
        """

        # ==================================================
        # 1️⃣ LSTM编码时序信息
        # ==================================================
        out, _ = self.lstm(x)
        # out shape: (B, T=30, H=128)
        # 每个时间步都有一个隐藏状态表示

        # ==================================================
        # 2️⃣ ⭐时序池化（mean pooling）
        # ==================================================
        feat_seq = out.mean(dim=1)
        # shape: (B, 128)
        # 含义：对30帧做平均 → 提取整体动作语义
        # 优点：
        #   - 比 last-step 更稳定
        #   - 不依赖最后一帧（减少噪声影响）

        # ==================================================
        # 3️⃣ 特征压缩 + 稳定化
        # ==================================================
        feat = self.feature_proj(feat_seq)
        # shape: (B, 64)
        # 含义：
        #   - 降维
        #   - LayerNorm稳定分布
        #   - dropout防过拟合

        # ==================================================
        # 4️⃣ 多任务输出
        # ==================================================

        # 风险分类（logits）
        risk = self.risk_head(feat)
        # shape: (B, 1)
        # 注意：这里不是概率，是logits

        # 跌倒时间回归
        time = self.time_head(feat)
        # shape: (B, 1)
        # 直接输出连续值（如秒/帧数）

        return risk, time