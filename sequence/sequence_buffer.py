from collections import deque
import numpy as np




class SequenceBufferV3:
    """
    工业级跌倒趋势缓存器（v3）

    支持任意窗口长度，未满时按实际帧数返回。

    输出：
        raw: (T, 57)
        vel: (T, 57)
        acc: (T, 57)
    """

    def __init__(self, seq_len=90):
        self.seq_len = seq_len
        self.buffers = {}

    # ==================================================
    # 更新
    # ==================================================
    def update(self, person_id, feature):
        """
        feature: (57,)
        """

        feature = np.asarray(feature, dtype=np.float32)

        # ❗异常过滤（工业必须）
        if feature.ndim != 1:
            return

        if np.isnan(feature).any():
            return

        if person_id not in self.buffers:
            self.buffers[person_id] = deque(maxlen=self.seq_len)

        self.buffers[person_id].append(feature)

    # ==================================================
    # 当前缓存的帧数
    # ==================================================
    def buffer_len(self, person_id):
        if person_id not in self.buffers:
            return 0
        return len(self.buffers[person_id])

    # ==================================================
    # 是否已满（达到窗口大小）
    # ==================================================
    def is_ready(self, person_id):
        return self.buffer_len(person_id) >= 1

    # ==================================================
    # 原始序列（未满时按实际帧数返回）
    # ==================================================
    def get_sequence(self, person_id):
        n = self.buffer_len(person_id)
        if n == 0:
            return None

        return np.asarray(self.buffers[person_id], dtype=np.float32)

    # ==================================================
    # 速度（核心趋势信号）
    # ==================================================
    def get_velocity(self, person_id):
        seq = self.get_sequence(person_id)
        if seq is None:
            return None

        vel = np.zeros_like(seq)
        if len(seq) >= 2:
            vel[1:] = seq[1:] - seq[:-1]

        return vel

    # ==================================================
    # 加速度（趋势加速变化）
    # ==================================================
    def get_acceleration(self, person_id):
        vel = self.get_velocity(person_id)
        if vel is None:
            return None

        acc = np.zeros_like(vel)
        if len(vel) >= 2:
            acc[1:] = vel[1:] - vel[:-1]

        return acc

    # ==================================================
    # 删除人
    # ==================================================
    def remove(self, person_id):
        if person_id in self.buffers:
            del self.buffers[person_id]

    def clear(self):
        self.buffers.clear()

    # ==================================================
    # 调试
    # ==================================================
    def __len__(self):
        return len(self.buffers)

    def __repr__(self):
        return f"SequenceBufferV3(persons={len(self.buffers)}, seq_len={self.seq_len})"


# 向后兼容别名
SequenceBuffer = SequenceBufferV3