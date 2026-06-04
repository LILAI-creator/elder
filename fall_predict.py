import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import deque
from ultralytics import YOLO

WINDOW = 30
DANGER_THRESHOLD = 0.8
WARNING_THRESHOLD = 0.4

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_baseline.pt")
NORM_PATH = os.path.join(MODEL_DIR, "norm_params.npz")
YOLO_MODEL = os.path.join(MODEL_DIR, "yolo11n-pose.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=34,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out



def load_lstm():
    model = C:\Users\R9000P\.conda\envs\yolo\python.exe D:\myproject\elder\realtime_demo.py --video "D:\my_datasets\Le2i\Le2i\Le2i\Home_01\Home_01\Videos\video (1).avi"().to(DEVICE)
    model.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def load_norm():
    data = np.load(NORM_PATH)
    return data["mean"], data["std"]


def extract_keypoints(results):
    if results.keypoints is None or len(results.keypoints) == 0:
        return None
    kpts = results.keypoints[0].data.cpu().numpy()
    if kpts.ndim == 3:
        kpts = kpts[0]
    if kpts.shape[0] < 17:
        return None
    xy = kpts[:17, :2]
    return xy.reshape(-1).astype(np.float32)


def predict(model, window_buf, mean, std):
    seq = np.array(window_buf, dtype=np.float32)
    seq = (seq - mean) / std
    x = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        return probs[0, 1].item()


def get_status(danger_prob):
    if danger_prob < WARNING_THRESHOLD:
        return "SAFE", (0, 255, 0)
    elif danger_prob < DANGER_THRESHOLD:
        return "DANGER", (0, 165, 255)
    else:
        return "FALL WARNING", (0, 0, 255)


def draw_overlay(frame, danger_prob, status, color, frame_idx=None):
    cv2.putText(
        frame, f"Danger: {danger_prob:.2f}", (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3,
    )
    cv2.putText(
        frame, status, (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3,
    )
    if frame_idx is not None:
        cv2.putText(
            frame, f"Frame: {frame_idx}", (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
