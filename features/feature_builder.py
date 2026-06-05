import numpy as np

class FeatureBuilderV3:
    """
    FeatureBuilderV3: 构建每帧人体关键点特征向量（v3标准）
    
    输入：
        kpts  - numpy array, (17,2)，每帧17个关键点的(x, y)坐标
        confs - numpy array, (17,)，对应每个关键点的置信度
    输出：
        feature - numpy array, (51,)，每帧flatten后的(x, y, conf)特征
    """

    def __init__(self):
        # 目前不需要初始化参数
        pass

    def build(self, kpts, confs):
        """
        构建单帧特征向量
        """

        # 将置信度从(17,)转为(17,1)列向量，方便和坐标拼接
        confs = confs.reshape(-1, 1)  # (17,1)

        # 将关键点坐标和置信度沿列拼接
        # kpts: (17,2), confs: (17,1) => feature: (17,3)
        feature = np.concatenate([kpts, confs], axis=1)  # (17,3)

        # 将每帧特征展平为一维向量
        # 原本每帧是17个点，每点3个值 => 17*3=51维
        return feature.reshape(-1)  # (51,)