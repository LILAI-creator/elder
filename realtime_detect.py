"""
视频实时跌倒检测演示

流程: 视频帧 → YOLO Pose(17关键点) → SequenceBuffer(30帧缓存) → 拼接[pos,vel,acc]=(30,102) → LSTM → risk/time/label
"""

import os
import cv2
import numpy as np
from pose.pose_extractor import PoseExtractor
from sequence.sequence_buffer import SequenceBufferV3
from classifier.lstm_classifier import LSTMClassifier

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
POSE_MODEL = os.path.join(MODEL_DIR, "yolo11n-pose.pt")
LSTM_MODEL = os.path.join(MODEL_DIR, "lstm_multitask.pt")
NORM_PARAMS = os.path.join(MODEL_DIR, "norm_params.npz")

WINDOW_SIZE = 30
RISK_THRESHOLD = 0.5


def build_102_feature(buffer, person_id):
    raw = buffer.get_sequence(person_id)
    if raw is None:
        return None
    vel = buffer.get_velocity(person_id)
    acc = buffer.get_acceleration(person_id)
    return np.concatenate([raw, vel, acc], axis=1)


def draw_result(frame, bbox, result, person_id):
    x1, y1, x2, y2 = map(int, bbox)
    risk = result["risk"]
    time_val = result["time"]
    label = result["label"]

    if label == 1:
        color = (0, 0, 255)
        status = "FALL"
    elif risk > 0.03:
        color = (0, 165, 255)
        status = "WARNING"
    else:
        color = (0, 255, 0)
        status = "SAFE"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"ID:{person_id} {status}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"risk:{risk:.3f} time:{time_val:.1f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def main(video_path=None):
    pose_ext = PoseExtractor(model_path=POSE_MODEL)
    buffer = SequenceBufferV3(seq_len=WINDOW_SIZE)
    classifier = LSTMClassifier(model_path=LSTM_MODEL, norm_path=NORM_PARAMS)

    if video_path is None:
        cap = cv2.VideoCapture(0)
        source_name = "Camera"
    else:
        cap = cv2.VideoCapture(video_path)
        source_name = os.path.basename(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open {source_name}")
        return

    print(f"Source: {source_name}")
    print("Press 'q' to quit\n")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        persons = pose_ext.extract(frame)

        for person in persons:
            kpts = person["keypoints"]
            bbox = person["bbox"]
            feature = kpts.reshape(-1)

            person_id = 0
            buffer.update(person_id, feature)

            if buffer.is_ready(person_id):
                seq_102 = build_102_feature(buffer, person_id)
                if seq_102 is not None:
                    result = classifier.predict(seq_102)
                    frame = draw_result(frame, bbox, result, person_id)

        cv2.putText(frame, f"Frame: {frame_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Fall Detection", frame)

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = max(1, int(1000 / fps)) if fps > 0 else 33
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else None
    main(video)