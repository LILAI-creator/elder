"""
fall_detector.py

功能：
    跌倒检测流水线

流程：

    Frame
      ↓
    PoseExtractor
      ↓
    PersonTracker
      ↓
    FeatureBuilder
      ↓
    SequenceBuffer
      ↓
    LSTMClassifier
      ↓
    Fall Result
"""

from pose.pose_extractor import PoseExtractor
from tracker.person_tracker import PersonTracker
from features.feature_builder import FeatureBuilder
from sequence.sequence_buffer import SequenceBuffer
from classifier.lstm_classifier import LSTMClassifier
from pipeline.risk_engine import RiskEngine

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
        frame : ndarray

        Returns
        -------
        list
        """

        # ==================================================
        # 1. Pose
        # ==================================================
        persons = self.pose_extractor.extract(frame)

        # ==================================================
        # 2. Tracking（🔥关键修复点）
        # ==================================================
        tracks, removed_ids = self.tracker.update(persons)

        # 清理已消失的ID
        for rid in removed_ids:
            self.buffer.remove(rid)

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
            # Feature
            # -----------------------------
            feature = FeatureBuilder.build(keypoints)

            # -----------------------------
            # Sequence Buffer
            # -----------------------------
            self.buffer.update(person_id, feature)

            # 序列不足30帧
            if not self.buffer.is_ready(person_id):

                results.append({
                    "id": person_id,
                    "bbox": bbox,
                    "safe": None,
                    "danger": None,
                    "label": None
                })

                continue

            # -----------------------------
            # 获取序列
            # -----------------------------
            sequence = self.buffer.get_sequence(person_id)

            # -----------------------------
            # LSTM预测
            # -----------------------------
            pred = self.classifier.predict(sequence)

            risk = self.risk_engine.update(
                person_id,
                pred["danger"]
            )

            results.append({
                "id": person_id,
                "bbox": bbox,
                "safe": pred["safe"],
                "danger": pred["danger"],
                "label": pred["label"],

                # 🔥 新增
                "risk_score": risk["risk_score"],
                "state": risk["state"]
            })

        return results

    # ==================================================
    # 重置系统
    # ==================================================
    def reset(self):
        """
        清空缓存
        """
        self.buffer.clear()