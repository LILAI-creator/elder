"""
用YOLO11n-Pose重新提取所有训练视频的keypoints，然后构建数据集并训练
解决训练/推理keypoints分布不一致的问题
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from glob import glob

ANNOTATION_BASE = r"D:\my_datasets\Le2i\Le2i\Le2i"
KEYPOINTS_OUTPUT = r"D:\my_datasets\Le2i\yolo_keypoints"
MODEL_PATH = r"D:\myproject\elder\models\yolo11n-pose.pt"

WINDOW_SIZE = 30
STRIDE = 5
NORMAL_TIME = 999

SCENES = {
    "Coffee_room_01": {
        "video_dir": os.path.join(ANNOTATION_BASE, "Coffee_room_01", "Coffee_room_01", "Videos"),
        "anno_dir": os.path.join(ANNOTATION_BASE, "Coffee_room_01", "Coffee_room_01", "Annotation_files"),
    },
    "Coffee_room_02": {
        "video_dir": os.path.join(ANNOTATION_BASE, "Coffee_room_02", "Coffee_room_02", "Videos"),
        "anno_dir": os.path.join(ANNOTATION_BASE, "Coffee_room_02", "Coffee_room_02", "Annotations_files"),
    },
    "Home_01": {
        "video_dir": os.path.join(ANNOTATION_BASE, "Home_01", "Home_01", "Videos"),
        "anno_dir": os.path.join(ANNOTATION_BASE, "Home_01", "Home_01", "Annotation_files"),
    },
    "Home_02": {
        "video_dir": os.path.join(ANNOTATION_BASE, "Home_02", "Home_02", "Videos"),
        "anno_dir": os.path.join(ANNOTATION_BASE, "Home_02", "Home_02", "Annotations_files"),
    },
}


def parse_annotation(annotation_path):
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    try:
        fall_start = int(lines[0].strip())
        fall_end = int(lines[1].strip())
    except ValueError:
        return None, None
    if fall_start <= 0 or fall_end <= 0:
        return None, None
    return fall_start, fall_end


def extract_keypoints_from_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_keypoints = np.zeros((total_frames, 17, 2), dtype=np.float32)
    all_confs = np.zeros((total_frames, 17), dtype=np.float32)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, device="0")
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kpts = results[0].keypoints
            xy = kpts.xy.cpu().numpy()
            if xy.ndim == 3 and xy.shape[1] == 17 and xy.shape[2] == 2:
                if kpts.conf is not None:
                    conf = kpts.conf.cpu().numpy()
                    best_idx = conf.sum(axis=1).argmax()
                else:
                    best_idx = 0
                all_keypoints[frame_idx] = xy[best_idx]
                if kpts.conf is not None:
                    all_confs[frame_idx] = conf[best_idx]

        frame_idx += 1

    cap.release()
    if frame_idx < total_frames:
        all_keypoints = all_keypoints[:frame_idx]
        all_confs = all_confs[:frame_idx]
    return all_keypoints, all_confs


def compute_velocity(keypoints):
    velocity = np.zeros_like(keypoints)
    velocity[1:] = keypoints[1:] - keypoints[:-1]
    return velocity


def compute_acceleration(velocity):
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    return acceleration


def build_features(keypoints):
    T = keypoints.shape[0]
    velocity = compute_velocity(keypoints)
    acceleration = compute_acceleration(velocity)
    keypoints_flat = keypoints.reshape(T, 34)
    velocity_flat = velocity.reshape(T, 34)
    acceleration_flat = acceleration.reshape(T, 34)
    return np.concatenate([keypoints_flat, velocity_flat, acceleration_flat], axis=1)


def main():
    model = YOLO(MODEL_PATH)
    os.makedirs(KEYPOINTS_OUTPUT, exist_ok=True)

    all_samples = []
    stats = {"fall_process": 0, "fall_after": 0, "normal": 0, "skipped": 0}

    for scene_name, scene_info in SCENES.items():
        video_dir = scene_info["video_dir"]
        anno_dir = scene_info["anno_dir"]
        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".avi")])

        for vf in video_files:
            video_id = vf.replace("video (", "").replace(").avi", "")
            video_path = os.path.join(video_dir, vf)
            anno_file = os.path.join(anno_dir, f"video ({video_id}).txt")

            if not os.path.exists(anno_file):
                stats["skipped"] += 1
                continue

            fall_start, fall_end = parse_annotation(anno_file)
            is_fall = fall_start is not None

            kp_save_path = os.path.join(KEYPOINTS_OUTPUT, f"{scene_name}_video_{video_id}_keypoints.npy")
            conf_save_path = os.path.join(KEYPOINTS_OUTPUT, f"{scene_name}_video_{video_id}_confs.npy")

            if os.path.exists(kp_save_path):
                keypoints = np.load(kp_save_path)
                confs = np.load(conf_save_path)
                print(f"  Loaded cached: {scene_name} video {video_id}")
            else:
                print(f"  Extracting: {scene_name} video {video_id}")
                keypoints, confs = extract_keypoints_from_video(model, video_path)
                np.save(kp_save_path, keypoints)
                np.save(conf_save_path, confs)

            features = build_features(keypoints)
            T = features.shape[0]

            if not is_fall:
                for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
                    end = start + WINDOW_SIZE
                    all_samples.append({
                        "x": features[start:end].astype(np.float32),
                        "risk": np.array([0.0], dtype=np.float32),
                        "time": np.array([NORMAL_TIME], dtype=np.float32),
                    })
                stats["normal"] += len(range(0, T - WINDOW_SIZE + 1, STRIDE))
            else:
                pre_fall = fall_start - WINDOW_SIZE
                for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
                    end = start + WINDOW_SIZE
                    x = features[start:end]
                    last_frame_idx = end - 1

                    if last_frame_idx >= fall_end - 1:
                        risk = 1.0
                        time = 0
                        stats["fall_after"] += 1
                    elif last_frame_idx >= pre_fall:
                        time = max(0, fall_end - 1 - last_frame_idx)
                        risk = 1.0 / (1.0 + time)
                        stats["fall_process"] += 1
                    else:
                        risk = 0.0
                        time = NORMAL_TIME
                        stats["normal"] += 1

                    all_samples.append({
                        "x": x.astype(np.float32),
                        "risk": np.array([risk], dtype=np.float32),
                        "time": np.array([time], dtype=np.float32),
                    })

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"  Fall process: {stats['fall_process']}")
    print(f"  Fall after: {stats['fall_after']}")
    print(f"  Normal: {stats['normal']}")
    print(f"  Skipped: {stats['skipped']}")

    output_dir = r"D:\my_datasets\Le2i\processed"
    os.makedirs(output_dir, exist_ok=True)
    xs = np.stack([s["x"] for s in all_samples])
    risks = np.stack([s["risk"] for s in all_samples])
    times = np.stack([s["time"] for s in all_samples])

    np.save(os.path.join(output_dir, "x.npy"), xs)
    np.save(os.path.join(output_dir, "risk.npy"), risks)
    np.save(os.path.join(output_dir, "time.npy"), times)

    print(f"\nSaved to {output_dir}")
    print(f"  x.shape: {xs.shape}")
    print(f"  risk.shape: {risks.shape}")
    print(f"  time.shape: {times.shape}")


if __name__ == "__main__":
    main()