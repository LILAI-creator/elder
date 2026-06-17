"""
fall_detector.py

功能：
    跌倒检测流水线（多人实时）

流程：

    Frame
      ↓
    PoseExtractor
      ↓
    PersonTracker
      ↓
    FeatureBuilder  (57维)
      ↓
    SequenceBuffer  (30帧缓存)
      ↓
    拼接 [pos(57) + vel(57) + acc(57)] = (30,171)
      ↓
    LSTMClassifier  → {risk, time, label}
      ↓
    RiskEngine  → SAFE / WARNING / DANGER
"""

import numpy as np

from pose.pose_extractor import PoseExtractor
from tracker.person_tracker import PersonTracker
from features.feature_builder import FeatureBuilder
from sequence.sequence_buffer import SequenceBuffer
from classifier.lstm_classifier import LSTMClassifier
from pipeline.risk_engine import RiskEngine

# 当前 FeatureBuilder 输出维度
FEATURE_DIM = 57  # 51基础(17×3) + 6几何特征


def build_motion_feature(buffer, person_id):
    """
    拼接 [位置, 速度, 加速度] → (30, FEATURE_DIM*3)
    """
    raw = buffer.get_sequence(person_id)
    if raw is None:
        return None
    vel = buffer.get_velocity(person_id)
    acc = buffer.get_acceleration(person_id)
    return np.concatenate([raw, vel, acc], axis=1)


class FallDetector:
    """
    跌倒检测主流程
    """

    def __init__(
            self,
            pose_model_path,
            lstm_model_path,
            norm_path,
            seq_len=30
    ):

        # -----------------------------
        # Risk Engine
        # -----------------------------
        self.risk_engine = RiskEngine()

        # -----------------------------
        # Pose
        # -----------------------------
        self.pose_extractor = PoseExtractor(
            pose_model_path
        )

        # -----------------------------
        # Tracker
        # -----------------------------
        self.tracker = PersonTracker()

        # -----------------------------
        # Sequence Buffer
        # -----------------------------
        self.buffer = SequenceBuffer(
            seq_len=seq_len
        )

        # -----------------------------
        # LSTM Classifier
        # -----------------------------
        self.classifier = LSTMClassifier(
            model_path=lstm_model_path,
            norm_path=norm_path
        )

    # ==================================================
    # 单帧处理
    # ==================================================
    def process(self, frame):
        """
        Parameters
        ----------
        frame : ndarray  (H, W, 3)

        Returns
        -------
        list[dict]
        """

        # ==================================================
        # 1. Pose
        # ==================================================
        persons = self.pose_extractor.extract(frame)

        # ==================================================
        # 2. Tracking
        # ==================================================
        tracks, removed_ids = self.tracker.update(persons)

        # 清理已消失的ID
        for rid in removed_ids:
            self.buffer.remove(rid)
            self.risk_engine.reset(rid)

        results = []

        # ==================================================
        # 3. 遍历Track
        # ==================================================
        for track in tracks:

            # 防御式写法（避免异常结构）
            if not isinstance(track, dict):
                continue

            person_id = track["id"]
            bbox = track["bbox"]
            keypoints = track["keypoints"]

            # -----------------------------
            # Feature (57维)
            # -----------------------------
            feature = FeatureBuilder.build(keypoints)

            # -----------------------------
            # Sequence Buffer
            # -----------------------------
            self.buffer.update(person_id, feature)

            # 序列不足30帧 → 占位返回
            if not self.buffer.is_ready(person_id):

                results.append({
                    "id": person_id,
                    "bbox": bbox,
                    "risk": None,
                    "time": None,
                    "label": None,
                    "state": "SAFE"
                })

                continue

            # -----------------------------
            # 拼接运动特征 (30, 171)
            # -----------------------------
            sequence = build_motion_feature(self.buffer, person_id)

            # -----------------------------
            # LSTM预测
            # -----------------------------
            # 当前模型 (lstm_multitask.pt) 已用 171 维特征训练，
            # 与 FeatureBuilder 57 维 + 运动拼接 (30,171) 匹配。
            pred = self.classifier.predict(sequence)

            risk_result = self.risk_engine.update(
                person_id,
                pred["risk"]
            )

            results.append({
                "id": person_id,
                "bbox": bbox,
                "risk": pred["risk"],
                "time": pred["time"],
                "label": pred["label"],

                # RiskEngine 输出
                "risk_score": risk_result["risk_score"],
                "state": risk_result["state"]
            })

        return results

    # ==================================================
    # 重置系统
    # ==================================================
    def reset(self):
        """
        清空所有缓存
        """
        self.buffer.clear()
        self.tracker.reset()
        self.risk_engine.reset()
