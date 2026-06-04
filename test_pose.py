from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    if len(results[0].keypoints.xy):

        person = results[0].keypoints.xy[0]

        print(person.shape)

        feature = person.cpu().numpy().flatten()

        print(feature.shape)

    cv2.imshow("pose", results[0].plot())

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()