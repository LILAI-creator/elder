import numpy as np
from features.label_generator_v3 import LabelGeneratorV3


def main():

    gen = LabelGeneratorV3(fps=10, max_time=3.0)

    seq_len = 30

    # =========================
    # 模拟“跌倒发生在第25帧”
    # =========================
    fall_frame = 25

    risk, time_to_fall = gen.generate(
        seq_len=seq_len,
        fall_frame=fall_frame
    )

    print("📊 Label Test Result\n")

    print("risk_score:")
    print(np.round(risk, 3))

    print("\ntime_to_fall:")
    print(np.round(time_to_fall, 2))

    print("\n🔍 关键片段分析:")

    print("\n[最后10帧 risk]:")
    print(np.round(risk[-10:], 3))

    print("\n[最后10帧 time]:")
    print(np.round(time_to_fall[-10:], 2))


if __name__ == "__main__":
    main()