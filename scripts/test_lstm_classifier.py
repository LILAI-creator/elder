"""
test_lstm_classifier.py


验证
✔ 模型是否能正常加载
✔ norm_params是否能正常加载
✔ SequenceBuffer输出是否兼容
✔ LSTM推理是否正常
✔ 输出概率是否正常
"""

import os
import sys
import numpy as np

# 项目根目录
ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT)

from classifier.lstm_classifier import (
    LSTMClassifier
)


def main():

    model_path = os.path.join(
        ROOT,
        "models",
        "lstm_baseline.pt"
    )

    norm_path = os.path.join(
        ROOT,
        "models",
        "norm_params.npz"
    )

    print("=" * 50)
    print("Loading classifier...")
    print("=" * 50)

    classifier = LSTMClassifier(
        model_path=model_path,
        norm_path=norm_path
    )

    print()

    # ---------------------------------
    # 模拟SequenceBuffer输出
    # ---------------------------------

    sequence = np.random.rand(
        30,
        34
    ).astype(np.float32)

    print("=" * 50)
    print("Input")
    print("=" * 50)

    print(
        "sequence shape:",
        sequence.shape
    )

    print()

    # ---------------------------------
    # 推理
    # ---------------------------------

    result = classifier.predict(
        sequence
    )

    print("=" * 50)
    print("Output")
    print("=" * 50)

    print(
        f"Safe   : {result['safe']:.4f}"
    )

    print(
        f"Danger : {result['danger']:.4f}"
    )

    print(
        f"Label  : {result['label']}"
    )

    print()

    # ---------------------------------
    # 仅概率接口
    # ---------------------------------

    danger_prob = classifier.predict_proba(
        sequence
    )

    print(
        f"predict_proba() = "
        f"{danger_prob:.4f}"
    )

    # ---------------------------------
    # 仅标签接口
    # ---------------------------------

    label = classifier.predict_label(
        sequence
    )

    print(
        f"predict_label() = "
        f"{label}"
    )

    print()
    print("Test Finished")


if __name__ == "__main__":
    main()