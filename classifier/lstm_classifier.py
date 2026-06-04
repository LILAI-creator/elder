"""
lstm_classifier.py

功能：
    加载训练好的LSTM模型

输入：

    sequence
    shape=(30,34)

输出：

    {
        "safe": float,
        "danger": float,
        "label": int
    }
"""

import numpy as np
import torch
import torch.nn as nn


# =====================================================
# 与训练阶段完全一致
# =====================================================

class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=34,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(
            128,
            2
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


# =====================================================
# 推理器
# =====================================================

class LSTMClassifier:

    def __init__(
            self,
            model_path,
            norm_path,
            device=None):

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        # ---------------------------------
        # 加载模型
        # ---------------------------------

        self.model = LSTMModel()

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=device
            )
        )

        self.model.to(device)

        self.model.eval()

        # ---------------------------------
        # 加载归一化参数
        # ---------------------------------

        norm = np.load(norm_path)

        self.mean = norm["mean"].astype(
            np.float32
        )

        self.std = norm["std"].astype(
            np.float32
        )

        print(
            f"[Classifier] Model loaded:"
            f" {model_path}"
        )

        print(
            f"[Classifier] Norm loaded:"
            f" {norm_path}"
        )

    # =====================================================
    # 归一化
    # =====================================================

    def normalize(self, sequence):
        """
        sequence

        (30,34)
        """

        return (
            sequence - self.mean
        ) / self.std

    # =====================================================
    # 推理
    # =====================================================

    @torch.no_grad()
    def predict(self, sequence):

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        if sequence.shape != (30, 34):

            raise ValueError(
                f"Expected (30,34), "
                f"got {sequence.shape}"
            )

        # ---------------------------------
        # 与训练阶段一致
        # ---------------------------------

        sequence = self.normalize(
            sequence
        )

        x = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        # (30,34)
        # ->
        # (1,30,34)

        x = x.unsqueeze(0)

        x = x.to(
            self.device
        )

        logits = self.model(x)

        probs = torch.softmax(
            logits,
            dim=1
        )

        probs = probs.squeeze(0)

        safe_prob = float(
            probs[0].cpu()
        )

        danger_prob = float(
            probs[1].cpu()
        )

        label = int(
            danger_prob > safe_prob
        )

        return {
            "safe": safe_prob,
            "danger": danger_prob,
            "label": label
        }

    @torch.no_grad()
    def predict_proba(self, sequence):

        return self.predict(
            sequence
        )["danger"]

    @torch.no_grad()
    def predict_label(self, sequence):

        return self.predict(
            sequence
        )["label"]