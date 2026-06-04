import cv2

from pose_extractor import PoseExtractor

pose = PoseExtractor()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    persons = pose.extract(frame)

    for i, person in enumerate(persons):

        bbox = person["bbox"]

        x1, y1, x2, y2 = bbox.astype(int)

        # 画框
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # 显示编号
        cv2.putText(
            frame,
            f"Person {i}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # 画17个关键点
        for x, y in person["keypoints"]:

            cv2.circle(
                frame,
                (int(x), int(y)),
                3,
                (0, 0, 255),
                -1
            )

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()