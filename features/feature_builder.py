import numpy as np


class FeatureBuilder:

    @staticmethod
    def build(keypoints):
        """
        输入:
            keypoints (17,2)

        输出:
            feature (34,)
        """

        keypoints = np.asarray(keypoints, dtype=np.float32)

        if keypoints.shape != (17, 2):
            return np.zeros(34, dtype=np.float32)

        # ==================================================
        # 1. 原始坐标（最重要）
        # ==================================================
        raw = keypoints.reshape(-1)

        # ==================================================
        # 2. 人体中心点
        # ==================================================
        center = np.mean(keypoints, axis=0)

        # ==================================================
        # 3. bbox信息
        # ==================================================
        x_min, y_min = np.min(keypoints, axis=0)
        x_max, y_max = np.max(keypoints, axis=0)

        width = x_max - x_min
        height = y_max - y_min

        aspect_ratio = width / (height + 1e-6)

        # ==================================================
        # 4. 躯干倾斜（关键跌倒信号）
        # ==================================================
        try:
            shoulder = keypoints[5]
            hip = keypoints[11]
            torso_vec = shoulder - hip
            torso_angle = np.arctan2(torso_vec[1], torso_vec[0])
        except:
            torso_angle = 0.0

        # ==================================================
        # 5. 稳定性（关键！）
        # ==================================================
        keypoints_var = np.var(keypoints)

        # ==================================================
        # 6. 拼接额外特征（6维）
        # ==================================================
        extra = np.array([
            center[0],
            center[1],
            width,
            height,
            aspect_ratio,
            torso_angle
        ], dtype=np.float32)

        # ==================================================
        # 7. 输出（必须保持34维）
        # ==================================================
        # 用压缩方式：只取 raw 的前28维 + 6维extra = 34维
        raw_compressed = raw[:28]

        return np.concatenate([raw_compressed, extra])