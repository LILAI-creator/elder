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


def build_102_feature(buffer, person_id):
    raw = buffer.get_sequence(person_id)
    if raw is None:
        return None
    vel = buffer.get_velocity(person_id)
    acc = buffer.get_acceleration(person_id)
    return np.concatenate([raw, vel, acc], axis=1)


def main():
    video_path = r"D:\myproject\elder\test\video (1).avi"
    pose_ext = PoseExtractor(model_path=POSE_MODEL)
    buffer = SequenceBufferV3(seq_len=WINDOW_SIZE)
    classifier = LSTMClassifier(model_path=LSTM_MODEL, norm_path=NORM_PARAMS)

    norm = np.load(NORM_PARAMS)
    mean = norm["mean"]
    std = norm["std"]
    print(f"Norm mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"Norm std range: [{std.min():.2f}, {std.max():.2f}]")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    results_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        persons = pose_ext.extract(frame)
        if len(persons) == 0:
            continue

        person = persons[0]
        kpts = person["keypoints"]
        feature = kpts.reshape(-1)

        buffer.update(0, feature)

        if buffer.is_ready(0):
            seq_102 = build_102_feature(buffer, 0)
            if seq_102 is not None:
                result = classifier.predict(seq_102)
                results_log.append((frame_count, result["risk"], result["time"], result["label"]))

                if frame_count % 10 == 0 or result["risk"] > 0.3:
                    raw = buffer.get_sequence(0)
                    print(f"Frame {frame_count}: risk={result['risk']:.4f}, time={result['time']:.1f}, label={result['label']}")
                    print(f"  raw range: [{raw.min():.2f}, {raw.max():.2f}]")
                    normed = (seq_102 - mean) / std
                    print(f"  normed range: [{normed.min():.2f}, {normed.max():.2f}]")

    cap.release()

    if results_log:
        risks = [r[1] for r in results_log]
        labels = [r[3] for r in results_log]
        print(f"\nTotal predictions: {len(results_log)}")
        print(f"Risk range: [{min(risks):.4f}, {max(risks):.4f}]")
        print(f"Risk mean: {np.mean(risks):.4f}")
        print(f"FALL count: {sum(labels)}, SAFE count: {len(labels) - sum(labels)}")


if __name__ == "__main__":
    main()