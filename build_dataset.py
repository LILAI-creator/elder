import os
import numpy as np
from glob import glob

KEYPOINTS_DIR = r"D:\my_datasets\Le2i\le2i_keypoints\le2i_keypoints"
ANNOTATION_BASE = r"D:\my_datasets\Le2i\Le2i\Le2i"
OUTPUT_DIR = r"D:\my_datasets\Le2i\processed"

WINDOW_SIZE = 30
STRIDE = 5
NORMAL_TIME = 999

SCENE_ANNOTATION_DIR = {
    "Coffee_room_01": os.path.join(ANNOTATION_BASE, "Coffee_room_01", "Coffee_room_01", "Annotation_files"),
    "Coffee_room_02": os.path.join(ANNOTATION_BASE, "Coffee_room_02", "Coffee_room_02", "Annotations_files"),
    "Home_01": os.path.join(ANNOTATION_BASE, "Home_01", "Home_01", "Annotation_files"),
    "Home_02": os.path.join(ANNOTATION_BASE, "Home_02", "Home_02", "Annotations_files"),
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


def get_annotation(scene, video_id):
    anno_dir = SCENE_ANNOTATION_DIR.get(scene)
    if anno_dir is None or not os.path.isdir(anno_dir):
        return None, None
    anno_file = os.path.join(anno_dir, f"video ({video_id}).txt")
    if not os.path.exists(anno_file):
        return None, None
    return parse_annotation(anno_file)


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
    features = np.concatenate([keypoints_flat, velocity_flat, acceleration_flat], axis=1)
    return features


def main():
    all_samples = []
    stats = {"fall_process": 0, "fall_after": 0, "normal": 0}

    for category in ["fall", "normal"]:
        cat_dir = os.path.join(KEYPOINTS_DIR, category)
        kp_files = sorted(glob(os.path.join(cat_dir, "*_keypoints.npy")))

        for kp_path in kp_files:
            basename = os.path.basename(kp_path)
            name = basename.replace("_keypoints.npy", "")
            parts = name.split("_video_")
            scene = parts[0]
            video_id = int(parts[1])

            keypoints = np.load(kp_path)
            features = build_features(keypoints)
            T = features.shape[0]

            fall_start, fall_end = get_annotation(scene, video_id)
            is_fall = fall_start is not None

            if not is_fall:
                print(f"Processing {name} (normal)")
                for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
                    end = start + WINDOW_SIZE
                    all_samples.append({
                        "x": features[start:end].astype(np.float32),
                        "risk": np.array([0.0], dtype=np.float32),
                        "time": np.array([NORMAL_TIME], dtype=np.float32),
                    })
                stats["normal"] += len(range(0, T - WINDOW_SIZE + 1, STRIDE))
            else:
                print(f"Processing {name} (fall_start={fall_start}, fall_end={fall_end})")
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xs = np.stack([s["x"] for s in all_samples])
    risks = np.stack([s["risk"] for s in all_samples])
    times = np.stack([s["time"] for s in all_samples])

    np.save(os.path.join(OUTPUT_DIR, "x.npy"), xs)
    np.save(os.path.join(OUTPUT_DIR, "risk.npy"), risks)
    np.save(os.path.join(OUTPUT_DIR, "time.npy"), times)

    print(f"\nSaved to {OUTPUT_DIR}")
    print(f"  x.shape: {xs.shape}")
    print(f"  risk.shape: {risks.shape}")
    print(f"  time.shape: {times.shape}")
    print(f"  risk range: [{risks.min():.4f}, {risks.max():.4f}]")
    print(f"  time range: [{times.min():.1f}, {times.max():.1f}]")


if __name__ == "__main__":
    main()
