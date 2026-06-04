"""
sequence_buffer.py

功能：
    为每个Person维护固定长度动作序列

作用：

    YOLO Pose
          ↓
    PersonTracker
          ↓
    FeatureBuilder
          ↓
      (34,)
          ↓
    SequenceBuffer
          ↓
      (30,34)
          ↓
          LSTM

--------------------------------------------------

为什么需要SequenceBuffer？

LSTM属于时序模型。

不能只输入当前一帧：

    (34,)

而需要输入连续动作：

    (30,34)

因此需要为每个人维护：

    最近30帧关键点特征

--------------------------------------------------

数据结构：

{
    person_id : deque(maxlen=30)
}

例如：

{
    1 : deque([...]),
    2 : deque([...])
}

每个deque保存：

[
    feature1,
    feature2,
    ...
    feature30
]

其中：

feature.shape

=
(34,)

--------------------------------------------------

使用示例：

buffer = SequenceBuffer(seq_len=30)

buffer.update(
    person_id=1,
    feature=feature
)

if buffer.is_ready(1):

    sequence = buffer.get_sequence(1)

    # sequence.shape
    # (30,34)

--------------------------------------------------

注意：

当PersonTracker删除某个Track时：

    removed_ids

需要同步：

    buffer.remove(person_id)

否则缓存会持续增长。
"""

from collections import deque
import numpy as np


class SequenceBuffer:
    """
    多人动作序列缓存器

    为每个Person ID维护一个固定长度队列。

    Person ID
         ↓
    deque(maxlen=30)
         ↓
      (30,34)
    """

    def __init__(self, seq_len=30):
        """
        Parameters
        ----------
        seq_len : int

            LSTM输入序列长度

            默认：
                30
        """

        # 序列长度
        self.seq_len = seq_len

        # 所有人的缓存

        # {
        #     person_id : deque(...)
        # }
        self.buffers = {}

    # ==================================================
    # 更新缓存
    # ==================================================

    def update(self, person_id, feature):
        """
        添加一帧特征

        Parameters
        ----------
        person_id : int

            人员ID

        feature : ndarray

            shape:
                (34,)
        """

        feature = np.asarray(
            feature,
            dtype=np.float32
        )

        # 检查输入维度

        if feature.shape != (34,):
            raise ValueError(
                f"Expected feature shape (34,), "
                f"got {feature.shape}"
            )

        # 新用户

        if person_id not in self.buffers:

            self.buffers[person_id] = deque(
                maxlen=self.seq_len
            )

        # 添加最新特征

        self.buffers[person_id].append(
            feature
        )

    # ==================================================
    # 查询状态
    # ==================================================

    def is_ready(self, person_id):
        """
        判断是否已达到LSTM输入长度

        Parameters
        ----------
        person_id : int

        Returns
        -------
        bool

            True:
                已有30帧

            False:
                不足30帧
        """

        if person_id not in self.buffers:
            return False

        return (
            len(self.buffers[person_id])
            == self.seq_len
        )

    def length(self, person_id):
        """
        获取当前缓存长度

        Parameters
        ----------
        person_id : int

        Returns
        -------
        int
        """

        if person_id not in self.buffers:
            return 0

        return len(
            self.buffers[person_id]
        )

    # ==================================================
    # 获取序列
    # ==================================================

    def get_sequence(self, person_id):
        """
        获取LSTM输入序列

        Parameters
        ----------
        person_id : int

        Returns
        -------
        ndarray

            shape:
                (30,34)

        或

        None
        """

        if not self.is_ready(person_id):
            return None

        return np.asarray(
            self.buffers[person_id],
            dtype=np.float32
        )

    # ==================================================
    # 删除缓存
    # ==================================================

    def remove(self, person_id):
        """
        删除指定人员缓存

        使用场景：

        PersonTracker发现目标消失：

            removed_ids

        同步调用：

            buffer.remove(person_id)
        """

        if person_id in self.buffers:
            del self.buffers[person_id]

    def clear(self):
        """
        清空所有缓存
        """

        self.buffers.clear()

    # ==================================================
    # 调试接口
    # ==================================================

    def get_ids(self):
        """
        返回当前所有Person ID

        Returns
        -------
        list
        """

        return list(
            self.buffers.keys()
        )

    def has_person(self, person_id):
        """
        判断是否存在指定ID
        """

        return (
            person_id in self.buffers
        )

    # ==================================================
    # Python内置接口
    # ==================================================

    def __contains__(self, person_id):
        """
        支持：

            if 1 in buffer:
                ...
        """

        return (
            person_id in self.buffers
        )

    def __len__(self):
        """
        返回当前缓存人数

        示例：

            len(buffer)
        """

        return len(
            self.buffers
        )

    def __repr__(self):
        """
        打印对象信息
        """

        return (
            f"SequenceBuffer("
            f"persons={len(self.buffers)}, "
            f"seq_len={self.seq_len})"
        )