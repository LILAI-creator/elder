import numpy as np

k = np.load("D:/my_datasets/Le2i/le2i_keypoints/le2i_keypoints/fall/Coffee_room_01_video_10_keypoints.npy")
c = np.load("D:/my_datasets/Le2i/le2i_keypoints/le2i_keypoints/fall/Coffee_room_01_video_10_confs.npy")

print("keypoints:", k.shape)
print("confs:", c.shape)

print("sample k[0]:")
print(k[0])