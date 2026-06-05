"""
lstm_classifier.py

输入:
    sequence shape=(30, 102)
    102 = 17*2坐标 + 17*2速度 + 17*2加速度

输出:
    {
        "risk": float,   # 跌倒概率 [0, 1]
        "time": float,   # 距离跌倒完成时间（帧数）
        "label": int     # 0=安全, 1=跌倒
    }
"""

import numpy as np
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=102, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.risk_head = nn.Linear(hidden_size, 1)
        self.time_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        risk = torch.sigmoid(self.risk_head(out))
        time = torch.relu(self.time_head(out))
        return risk.squeeze(-1), time.squeeze(-1)


class LSTMClassifier:
    def __init__(self, model_path, norm_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = LSTMModel()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        norm = np.load(norm_path)
        self.mean = norm["mean"].astype(np.float32)
        self.std = norm["std"].astype(np.float32)

        print(f"[Classifier] Model loaded: {model_path}")
        print(f"[Classifier] Norm loaded: {norm_path}")

    def normalize(self, sequence):
        return (sequence - self.mean) / self.std

    @torch.no_grad()
    def predict(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.shape != (30, 102):
            raise ValueError(f"Expected (30, 102), got {sequence.shape}")

        sequence = self.normalize(sequence)
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        risk, time = self.model(x)
        risk_val = float(risk.cpu())
        time_val = float(time.cpu())
        label = int(risk_val > 0.5)

        return {
            "risk": risk_val,
            "time": time_val,
            "label": label,
        }

    @torch.no_grad()
    def predict_risk(self, sequence):
        return self.predict(sequence)["risk"]

    @torch.no_grad()
    def predict_time(self, sequence):
        return self.predict(sequence)["time"]

    @torch.no_grad()
    def predict_label(self, sequence):
        return self.predict(sequence)["label"]
