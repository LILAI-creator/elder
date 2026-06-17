import numpy as np


class FeatureBuilder:
    """
    单帧人体关键点特征构建器

    输入:
        kpts  - (17,2) 17个COCO关键点的(x,y)坐标
        confs - (17,)  每个关键点的置信度，可选

    输出:
        feature - (57,) = 51维基础特征 + 6维几何特征
          基础: 17×(x, y, conf)
          几何: 重心高度, 躯干倾斜角, 左膝角, 右膝角, 双脚距离, 人体高度
    """

    @staticmethod
    def _calc_angle(a, b, c):

        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0

        cos_theta = np.dot(ba, bc) / (
            norm_ba * norm_bc
        )

        cos_theta = np.clip(
            cos_theta,
            -1.0,
            1.0
        )

        angle = np.degrees(
            np.arccos(cos_theta)
        )

        return angle

    @staticmethod
    def build(kpts, confs=None):
        """
        构建单帧特征向量

        Parameters
        ----------
        kpts : ndarray (17,2)
            17个关键点坐标
        confs : ndarray (17,) or None
            关键点置信度，为None时默认全1

        Returns
        -------
        feature : ndarray (57,)
        """
        if confs is None:
            confs = np.ones(17, dtype=np.float32)

        confs = confs.reshape(-1, 1)

        feature = np.concatenate(
            [kpts, confs],
            axis=1
        ).reshape(-1)

        # ==========================
        # 重心高度
        # ==========================

        left_hip = kpts[11]
        right_hip = kpts[12]

        hip_center = (
            left_hip + right_hip
        ) / 2

        cg_y = hip_center[1]

        # ==========================
        # 身体倾斜角
        # ==========================

        left_shoulder = kpts[5]
        right_shoulder = kpts[6]

        shoulder_center = (
            left_shoulder +
            right_shoulder
        ) / 2

        dx = (
            shoulder_center[0]
            - hip_center[0]
        )

        dy = (
            shoulder_center[1]
            - hip_center[1]
        )

        torso_angle = np.degrees(
            np.arctan2(dx, -dy)
        )

        # ==========================
        # 左膝角
        # ==========================

        left_knee_angle = FeatureBuilder._calc_angle(
            left_hip,
            kpts[13],
            kpts[15]
        )

        # ==========================
        # 右膝角
        # ==========================

        right_knee_angle = FeatureBuilder._calc_angle(
            right_hip,
            kpts[14],
            kpts[16]
        )

        # ==========================
        # 双脚距离
        # ==========================

        foot_distance = np.linalg.norm(
            kpts[15] - kpts[16]
        )

        # ==========================
        # 人体高度
        # ==========================

        nose = kpts[0]

        ankle_center = (
            kpts[15] +
            kpts[16]
        ) / 2

        body_height = np.linalg.norm(
            nose - ankle_center
        )

        extra = np.array(
            [
                cg_y,
                torso_angle,
                left_knee_angle,
                right_knee_angle,
                foot_distance,
                body_height
            ],
            dtype=np.float32
        )

        return np.concatenate(
            [
                feature,
                extra
            ]
        )