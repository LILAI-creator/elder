"""
test_fall_detector.py

功能：
    测试 FallDetector 全流程

不依赖摄像头：
    使用随机 frame + YOLO-Pose（或模拟Pose）

验证：

    1. pipeline是否能跑通
    2. Sequence是否正常累积
    3. LSTM是否能输出结果
"""

import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(ROOT)

from pipeline.fall_detector import FallDetector


# =====================================================
# 简单Mock Frame（避免必须摄像头）
# =====================================================

def get_fake_frame():
    """
    生成假frame（仅用于测试pipeline是否通）
    """
    return np.zeros((480, 640, 3), dtype=np.uint8)


# =====================================================
# 主测试
# =====================================================

def main():

    print("=" * 60)
    print("FallDetector Pipeline Test")
    print("=" * 60)

    detector = FallDetector(
        pose_model_path=os.path.join(
            ROOT,
            "models",
            "yolo11n-pose.pt"
        ),
        lstm_model_path=os.path.join(
            ROOT,
            "models",
            "lstm_baseline.pt"
        ),
        norm_path=os.path.join(
            ROOT,
            "models",
            "norm_params.npz"
        )
    )

    # =================================================
    # 连续喂帧（模拟视频流）
    # =================================================

    for i in range(50):

        frame = get_fake_frame()

        results = detector.process(frame)

        print(f"\nFrame {i}")

        for r in results:

            print(
                f"ID={r['id']} "
                f"risk={r.get('risk')} "
                f"time={r.get('time')} "
                f"label={r.get('label')} "
                f"state={r.get('state')}"
            )

    print("\nTest Finished")


if __name__ == "__main__":
    main()