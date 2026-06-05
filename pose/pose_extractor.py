"""
pose_extractor.py

功能：
    使用 YOLO-Pose 提取人体关键点

输入：
    一张图片(frame)

输出：
    [
        {
            "bbox": ndarray(4,),
            "keypoints": ndarray(17,2),
            "score": float
        }
    ]
"""

from ultralytics import YOLO
import numpy as np

modelPath = "./models/yolo11n-pose.pt"

class PoseExtractor:
    """
    人体姿态提取器
    """

    def __init__(self, model_path=modelPath):

        # 加载YOLO Pose模型
        self.model = YOLO(model_path)

    def extract(self, frame):
        """
        提取当前画面所有人的关键点

        Parameters
        ----------
        frame : ndarray
            OpenCV读取的图片

        Returns
        -------
        persons : list
        """

        persons = []

        # 推理
        results = self.model(frame, verbose=False)

        result = results[0]

        # 没有人
        if result.boxes is None:
            return persons

        # bbox
        boxes = result.boxes.xyxy.cpu().numpy()

        # 检测置信度
        scores = result.boxes.conf.cpu().numpy()

        # 17关键点
        keypoints = result.keypoints.xy.cpu().numpy()

        for bbox, score, kpts in zip(
                boxes,
                scores,
                keypoints):

            person = {
                "bbox": bbox,
                "keypoints": kpts,
                "score": float(score)
            }

            persons.append(person)

        return persons

    @staticmethod
    def build_feature(person):
        """
        把单个人的keypoints转成模型输入
        """

        kpts = person["keypoints"]  # (17,2)

        # 拉平
        feature = kpts.reshape(-1)  # (34,)

        return feature