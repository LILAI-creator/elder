import os
import re
import numpy as np

WINDOW = 30
PREFALL = 25

DATASET_ROOT = "D:/my_datasets/Le2i"
LE2I_ROOT = os.path.join(DATASET_ROOT, "Le2i", "Le2i")
KP_ROOT = os.path.join(DATASET_ROOT, "le2i_keypoints", "le2i_keypoints")

FALL_DIR = os.path.join(KP_ROOT, "fall")
NORMAL_DIR = os.path.join(KP_ROOT, "normal")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


def read_fall_annotation(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    fall_start = int(lines[0].strip())
    fall_end = int(lines[1].strip())
    return fall_start, fall_end


def build_fall_video(kp_path, fall_start, fall_end):
    kp = np.load(kp_path)
    T = len(kp)
    kp = kp.reshape(T, -1)

    labels = np.zeros(T, dtype=np.int64)
    danger_start = max(0, fall_start - PREFALL)
    labels[danger_start:] = 1

    X = []
    Y = []
    for i in range(T - WINDOW + 1):
        seq = kp[i : i + WINDOW]
        label = labels[i + WINDOW - 1]
        X.append(seq)
        Y.append(label)

    return X, Y


def build_normal_video(kp_path):
    kp = np.load(kp_path)
    T = len(kp)
    kp = kp.reshape(T, -1)

    X = []
    Y = []
    for i in range(T - WINDOW + 1):
        seq = kp[i : i + WINDOW]
        X.append(seq)
        Y.append(0)

    return X, Y


def parse_keypoint_filename(filename):
    m = re.match(r"(.+)_video_(\d+)_keypoints\.npy", filename)
    if m is None:
        return None, None
    scene = m.group(1)
    video_id = int(m.group(2))
    return scene, video_id


def find_annotation_txt(scene, video_id):
    for dirname in ("Annotation_files", "Annotations_files"):
        txt_path = os.path.join(
            LE2I_ROOT, scene, scene, dirname, f"video ({video_id}).txt"
        )
        if os.path.exists(txt_path):
            return txt_path
    return os.path.join(
        LE2I_ROOT, scene, scene, "Annotation_files", f"video ({video_id}).txt"
    )


def main():
    all_X = []
    all_Y = []

    # --- Fall videos ---
    fall_count = 0
    fall_skip = 0
    for file in sorted(os.listdir(FALL_DIR)):
        if not file.endswith("_keypoints.npy"):
            continue

        scene, video_id = parse_keypoint_filename(file)
        if scene is None:
            print(f"  [SKIP] cannot parse: {file}")
            fall_skip += 1
            continue

        txt_path = find_annotation_txt(scene, video_id)
        if not os.path.exists(txt_path):
            print(f"  [SKIP] annotation not found: {txt_path}")
            fall_skip += 1
            continue

        fall_start, fall_end = read_fall_annotation(txt_path)
        kp_path = os.path.join(FALL_DIR, file)
        X, Y = build_fall_video(kp_path, fall_start, fall_end)

        all_X.extend(X)
        all_Y.extend(Y)
        fall_count += 1
        print(f"  [FALL] {file}: fall_start={fall_start}, windows={len(X)}")

    print(f"\nFall videos processed: {fall_count}, skipped: {fall_skip}")

    # --- Normal videos ---
    normal_count = 0
    normal_skip = 0
    for file in sorted(os.listdir(NORMAL_DIR)):
        if not file.endswith("_keypoints.npy"):
            continue

        kp_path = os.path.join(NORMAL_DIR, file)
        X, Y = build_normal_video(kp_path)

        all_X.extend(X)
        all_Y.extend(Y)
        normal_count += 1
        print(f"  [NORMAL] {file}: windows={len(X)}")

    print(f"\nNormal videos processed: {normal_count}, skipped: {normal_skip}")

    # --- Build arrays ---
    X = np.array(all_X, dtype=np.float32)
    Y = np.array(all_Y, dtype=np.int64)

    print(f"\nX.shape = {X.shape}")
    print(f"Y.shape = {Y.shape}")

    unique, counts = np.unique(Y, return_counts=True)
    label_names = {0: "Safe", 1: "Danger"}
    for u, c in zip(unique, counts):
        print(f"  {label_names.get(u, u)}: {c}")

    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "Y.npy"), Y)
    print(f"\nSaved to {OUTPUT_DIR}/X.npy and {OUTPUT_DIR}/Y.npy")


if __name__ == "__main__":
    main()
