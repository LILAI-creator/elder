"""
test_tracker.py

功能：
    测试 PersonTracker

流程：

摄像头
    ↓
PoseExtractor
    ↓
PersonTracker
    ↓
显示ID和BBox
"""

import cv2

from pose.pose_extractor import PoseExtractor
from tracker.person_tracker import PersonTracker


def main():

    # ----------------------------------
    # 初始化
    # ----------------------------------

    pose = PoseExtractor()

    tracker = PersonTracker(
        max_missing=30,
        distance_threshold=100
    )

    # 摄像头
    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------
        # 提取人体
        # ----------------------------------

        persons = pose.extract(frame)

        # ----------------------------------
        # 跟踪
        # ----------------------------------

        tracks = tracker.update(persons)

        # ----------------------------------
        # 绘制结果
        # ----------------------------------

        for track in tracks:

            track_id = track["id"]

            bbox = track["bbox"]

            x1, y1, x2, y2 = bbox.astype(int)

            # 画框
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # 显示ID
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ----------------------------------
        # 显示人数
        # ----------------------------------

        cv2.putText(
            frame,
            f"Persons: {len(tracks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # ----------------------------------
        # 显示画面
        # ----------------------------------

        cv2.imshow(
            "Tracker Test",
            frame
        )

        key = cv2.waitKey(1)

        # ESC退出
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()