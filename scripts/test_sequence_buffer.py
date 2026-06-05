import numpy as np
import time

from sequence.sequence_buffer import SequenceBufferV3


def generate_fake_feature(step):
    """
    模拟34维人体特征
    让数据有“运动趋势”
    """

    base = np.linspace(100, 200, 34)

    noise = np.random.randn(34) * 0.5

    # 模拟人体逐渐下移（跌倒趋势）
    drift = step * 0.3

    return base + noise - drift


def main():

    buffer = SequenceBufferV3(seq_len=30)

    person_id = 1

    print("🚀 开始填充30帧数据...\n")

    # 模拟30帧
    for i in range(30):

        feature = generate_fake_feature(i)

        buffer.update(person_id, feature)

        print(f"frame {i+1} added")

        time.sleep(0.05)

    print("\n✅ 是否ready:", buffer.is_ready(person_id))

    # =========================
    # 获取数据
    # =========================
    seq = buffer.get_sequence(person_id)
    vel = buffer.get_velocity(person_id)
    acc = buffer.get_acceleration(person_id)

    print("\n📊 输出检查：")

    print("raw shape:", seq.shape)   # (30,34)
    print("vel shape:", vel.shape)   # (30,34)
    print("acc shape:", acc.shape)   # (30,34)

    print("\n🔍 前3帧 velocity 示例：")
    print(vel[:3])

    print("\n🔍 前3帧 acceleration 示例：")
    print(acc[:3])


if __name__ == "__main__":
    main()