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
import  cv2
modelPath = "./models/yolo11n-pose.pt"

class PoseExtractor:
    """
    人体姿态提取器
    """

    def __init__(self, model_path=modelPath):

        # 加载YOLO Pose模型
        self.model = YOLO(model_path)


    def predict(self, frame):
        results = self.model(frame, verbose=False)
        
        #单张图片
        result = results[0] 

        print(result)
        print(result.keypoints.data)

        '''
        YOLO 姿态估计模型返回的 keypoints 坐标是绝对像素坐标，坐标原点位于图像的左上角。具体来说：
        原点 (0,0)：图像左上角。
        x 轴：水平向右，数值为像素列索引。
        y 轴：垂直向下，数值为像素行索引。
        单位：像素（pixel）。
        '''
        annotated_frame = result.plot() 
        cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
        cv2.imshow("Pose Estimation", annotated_frame)


        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
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
        #verbose=False 是 YOLO 模型推理时的一个参数，用于控制是否打印详细日志信息。
        results = self.model(frame, verbose=False)
        '''
        属性	类型	形状	描述
        result.keypoints	Keypoints	(N)	关键点。
        result.keypoints.data	torch.float32	(N,K,2/3)	x,y 外加可选的可见性/置信度。
        result.keypoints.xy	torch.float32	(N,K,2)	像素关键点。
        result.keypoints.xyn	torch.float32	(N,K,2)	归一化关键点。
        result.boxes	Boxes	(N)	实例框。
        '''

        result = results[0]

        # 没有人
        if result.boxes is None:
            return persons

        # bbox
        boxes = result.boxes.xyxy.cpu().numpy()

        # 检测置信度
        scores = result.boxes.conf.cpu().numpy()

        # 17关键点 (xy) + 置信度（keypoints 或 conf 可能为 None）
        if result.keypoints is None or result.keypoints.xy is None:
            return persons

        keypoints = result.keypoints.xy.cpu().numpy()
        keypoints_conf = result.keypoints.conf
        if keypoints_conf is not None:
            keypoints_conf = keypoints_conf.cpu().numpy()   # (N, 17)
        else:
            keypoints_conf = [None] * len(keypoints)         # fallback

        for bbox, score, kpts, confs in zip(
                boxes,
                scores,
                keypoints,
                keypoints_conf):

            person = {
                "bbox": bbox,
                "keypoints": kpts,
                "keypoints_conf": confs,
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


def main():
    pose = PoseExtractor()
    pose.predict("./image.png")

    pass

if __name__ == "__main__":
    main()