import cv2
import numpy as np
from pose.pose_extractor import PoseExtractor

# 初始化
extractor = PoseExtractor()

# 读取一张测试图片（你可以换成自己的）
img_path = "image.png"
frame = cv2.imread(img_path)

if frame is None:
    print("❌ 图片读取失败，请检查路径")
    exit()

# 1. 提取人体
persons = extractor.extract(frame)

print(f"检测到人数: {len(persons)}")

# 2. 遍历每个人
for i, p in enumerate(persons):

    # keypoints
    kpts = p["keypoints"]
    print(f"\n👤 人{i}")
    print("keypoints shape:", kpts.shape)  # (17,2)

    # 3. 转 feature
    feature = PoseExtractor.build_feature(p)

    print("feature shape:", feature.shape)  # (34,)
    print("feature 前5维:", feature[:5])

# 4. 检查是否正常
if len(persons) == 0:
    print("⚠️ 没检测到人，请换图片")