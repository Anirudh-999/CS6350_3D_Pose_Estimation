import os
from main import run_pipeline
from typing import Dict
import numpy as np

LEFT_IMG_PATH  = "../images/img2.png"      # path to left image
RIGHT_IMG_PATH = "../images/img1.png"      # path to right image

# LEFT_IMG_PATH  = "../images/img3.jpeg"      # path to left image
# RIGHT_IMG_PATH = "../images/img4.jpeg"      # path to right image

CALIB_PATH     = "calib.txt"      # optional KITTI-style calib (P0/P1 or K)
YOLO_WEIGHTS   = "yolov8l.pt"     # optional (if ultralytics installed)
OUTPUT_DIR     = "./output"       # where we save visualizations
USE_GPU        = True             # preference flag


path = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(path, '..', 'dataset', 'rgb')
rgb_files_path = os.path.join(path, '..', 'dataset', 'rgb.txt')
ground_path = os.path.join(path, '..', 'dataset', 'groundtruth.txt')

rgb_paths: Dict[str, Dict[str, list]] = {}

def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R

with open(rgb_files_path, 'r') as file:
    for line in file:
        timestamp_str, img_path = line.strip().split()
        img_path = os.path.join("../dataset", img_path)
        # print(img_path)
        # break
        timestamp = round(float(timestamp_str), 2)
        timestamp_key = f"{timestamp:.2f}"
        rgb_paths[timestamp_key] = {"image": img_path, "pose": None}

with open(ground_path, 'r') as gr:
    for line in gr:
        timestamp_str, tx, ty, tz, qx, qy, qz, qw = line.strip().split()
        R = quaternion_to_rotation_matrix([float(qw), float(qx), float(qy), float(qz)])
        timestamp = round(float(timestamp_str), 2)
        timestamp_key = f"{timestamp:.2f}"
        if timestamp_key in rgb_paths:
            rgb_paths[timestamp_key]["pose"] = [tx, ty, tz, R]

rgb_paths = {k: v for k, v in rgb_paths.items() if v["pose"] is not None}

# print(rgb_paths)

image_keys = list(rgb_paths.keys())
# ef run_pipeline(left_path=LEFT_IMG_PATH, right_path=RIGHT_IMG_PATH, calib_path=CALIB_PATH, yolo_weights=YOLO_WEIGHTS):
for i in range(1):
    run_pipeline(rgb_paths[image_keys[i]]["image"], rgb_paths[image_keys[i+1]]["image"], calib_path= None, yolo_weights= YOLO_WEIGHTS)
    print("#"*20)
    print("ground_truth :=", rgb_paths[image_keys[i]]["pose"])
    run_pipeline(rgb_paths[image_keys[i+1]]["image"], rgb_paths[image_keys[i]]["image"], calib_path= None, yolo_weights= YOLO_WEIGHTS)