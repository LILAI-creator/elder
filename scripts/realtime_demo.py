"""
realtime_demo.py

功能：
    跌倒检测实时演示（视频输入版）
"""

import os
import cv2
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from pipeline.fall_detector import FallDetector


# =====================================================
# 可视化工具
# =====================================================
def draw(frame, results):

    for r in results:

        x1, y1, x2, y2 = map(int, r["bbox"])

        state = r.get("state", "UNKNOWN")
        risk = r.get("risk_score", None)

        # -----------------------------
        # 颜色策略
        # -----------------------------
        if state == "DANGER":
            color = (0, 0, 255)
        elif state == "WARNING":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # -----------------------------
        # 文本（防 None 崩溃）
        # -----------------------------
        if risk is None:
            text = f"ID:{r['id']} {state}"
        else:
            text = f"ID:{r['id']} {state} {risk:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame


# =====================================================
# 主函数
# =====================================================
def main(video_path):

    print("=" * 60)
    print("Realtime Fall Detection Demo")
    print("=" * 60)

    detector = FallDetector(
        pose_model_path=os.path.join(ROOT, "models", "yolo11n-pose.pt"),
        lstm_model_path=os.path.join(ROOT, "models", "lstm_baseline.pt"),
        norm_path=os.path.join(ROOT, "models", "norm_params.npz")
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    speed_factor = 0.25  # 0.5 = 0.5倍速（慢一半）

    delay = int(1000 / fps / speed_factor)

    print(f"FPS: {fps}")

    # =================================================
    # 主循环
    # =================================================
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        results = detector.process(frame)

        frame = draw(frame, results)

        cv2.imshow("Fall Detection Demo", frame)

        # ⭐关键：按真实FPS播放
        key = cv2.waitKey(delay) & 0xFF

        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# =====================================================
# 入口
# =====================================================
if __name__ == "__main__":

    video_path = os.path.join(ROOT, "test", "video (1).avi")

    main(video_path)