import os
import argparse
import cv2
from collections import deque
from ultralytics import YOLO
from fall_predict import (
    DEVICE, WINDOW, DANGER_THRESHOLD, YOLO_MODEL,
    load_lstm, load_norm,
    extract_keypoints, predict,
    get_status, draw_overlay,
)


def run_on_source(source, lstm_model, pose_model, mean, std, save_path=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    window_buf = deque(maxlen=WINDOW)
    danger_prob = 0.0
    frame_idx = 0
    fall_frame = -1

    print(f"Source: {source}")
    print(f"Resolution: {w}x{h}, FPS: {fps:.1f}, Frames: {total_frames}")
    print("Running... Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = pose_model(frame, verbose=False)[0]
        feat = extract_keypoints(results)

        if feat is not None:
            window_buf.append(feat)

        if len(window_buf) == WINDOW:
            danger_prob = predict(lstm_model, window_buf, mean, std)

        status, color = get_status(danger_prob)

        if danger_prob >= DANGER_THRESHOLD and fall_frame == -1:
            fall_frame = frame_idx
            print(f"  [ALERT] Fall detected at frame {frame_idx}, prob={danger_prob:.4f}")

        draw_overlay(frame, danger_prob, status, color, frame_idx)

        if writer:
            writer.write(frame)

        cv2.imshow("Fall Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"\nSaved result to {save_path}")
    cv2.destroyAllWindows()

    if fall_frame > 0:
        print(f"\nFall detected at frame {fall_frame}")
        if total_frames > 0:
            print(f"Time: {fall_frame / fps:.2f}s / {total_frames / fps:.2f}s")
    else:
        print("\nNo fall detected in video.")


def main():
    parser = argparse.ArgumentParser(description="Fall Prediction Demo")
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to video file. If not set, uses webcam (camera 0).",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save annotated output video (only for --video mode).",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    print("Loading LSTM model...")
    lstm_model = load_lstm()
    mean, std = load_norm()
    print(f"Norm params: mean shape={mean.shape}, std shape={std.shape}")

    print("Loading YOLO-Pose model...")
    pose_model = YOLO(YOLO_MODEL)

    if args.video:
        save_path = args.save
        if save_path is None:
            name = os.path.splitext(os.path.basename(args.video))[0]
            save_path = os.path.join(os.path.dirname(__file__), "outputs", f"{name}_result.mp4")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        run_on_source(args.video, lstm_model, pose_model, mean, std, save_path)
    else:
        print("\nOpening camera (0)...")
        run_on_source(0, lstm_model, pose_model, mean, std)


if __name__ == "__main__":
    main()
