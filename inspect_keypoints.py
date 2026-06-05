import os
import numpy as np

KEYPOINTS_DIR = r"D:\my_datasets\Le2i\le2i_keypoints\le2i_keypoints"

for category in ["fall", "normal"]:
    cat_dir = os.path.join(KEYPOINTS_DIR, category)
    files = sorted([f for f in os.listdir(cat_dir) if f.endswith("_keypoints.npy")])
    print(f"\n{'='*60}")
    print(f"Category: {category} ({len(files)} videos)")
    print(f"{'='*60}")

    for kp_file in files:
        kp_path = os.path.join(cat_dir, kp_file)
        conf_file = kp_file.replace("_keypoints.npy", "_confs.npy")
        conf_path = os.path.join(cat_dir, conf_file)

        kp = np.load(kp_path)
        conf = np.load(conf_path)

        name = kp_file.replace("_keypoints.npy", "")
        nonzero_frames = np.count_nonzero(kp.sum(axis=(1, 2)))
        print(f"  {name}: shape={kp.shape}, conf_shape={conf.shape}, nonzero_frames={nonzero_frames}/{kp.shape[0]}")