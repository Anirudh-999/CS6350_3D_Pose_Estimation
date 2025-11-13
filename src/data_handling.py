import os
from main import run_pipeline
from typing import Dict
import numpy as np
from epipolar_geometry import evaluate_pose_accuracy
import cv2

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

import numpy as np

def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion [w, x, y, z] to a 3x3 rotation matrix.
    Assumes q is a 4-element array or list, with q[0]=w.
    """
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*(q2**2 + q3**2),   2*(q1*q2 - q0*q3),       2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),       1 - 2*(q1**2 + q3**2),   2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),       2*(q2*q3 + q0*q1),       1 - 2*(q1**2 + q2**2)]
    ])
    return R

def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (in degrees).
    Returns angles as [roll, pitch, yaw] in ZYX order (extrinsic).
    """
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # Roll
        y = np.arctan2(-R[2, 0], sy)      # Pitch
        z = np.arctan2(R[1, 0], R[0, 0])  # Yaw
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.degrees(np.array([x, y, z]))


with open(rgb_files_path, 'r') as file:
    for line in file:
        timestamp_str, img_path = line.strip().split()
        img_path = os.path.join("../dataset", img_path)
        # print(img_path)
        # break
        timestamp = round(float(timestamp_str), 2)
        timestamp_key = f"{timestamp:.2f}"
        rgb_paths[timestamp_key] = {"image": img_path, "pose": None}

# Update ground truth processing to include Euler angles
with open(ground_path, 'r') as gr:
    for line in gr:
        timestamp_str, tx, ty, tz, qx, qy, qz, qw = line.strip().split()
        R = quaternion_to_rotation_matrix([float(qw), float(qx), float(qy), float(qz)])
        euler_angles = rotation_matrix_to_euler_angles(R)
        timestamp = round(float(timestamp_str), 2)
        timestamp_key = f"{timestamp:.2f}"
        if timestamp_key in rgb_paths:
            rgb_paths[timestamp_key]["pose"] = [tx, ty, tz, R, euler_angles]

rgb_paths = {k: v for k, v in rgb_paths.items() if v["pose"] is not None}

# print(rgb_paths)

image_keys = list(rgb_paths.keys())
# ef run_pipeline(left_path=LEFT_IMG_PATH, right_path=RIGHT_IMG_PATH, calib_path=CALIB_PATH, yolo_weights=YOLO_WEIGHTS):
for i in range(5):
    gt_cur = rgb_paths[image_keys[i]]["pose"]
    gt_next = rgb_paths[image_keys[i+1]]["pose"]

    print(f"\n=== Image Pair {i} ===")
    print("Ground Truth:", gt_cur)

    # --- (1) Run Essential Matrix pipeline ---
    results = run_pipeline(
        left_path=rgb_paths[image_keys[i+1]]["image"],
        right_path=rgb_paths[image_keys[i]]["image"],
        calib_path=None,
        yolo_weights=YOLO_WEIGHTS,
        gt_pose_cur=gt_cur,
        gt_pose_next=gt_next
    )
    
    R_gt_next = np.array(gt_next[3])
    R_gt_cur = np.array(gt_cur[3])
    
    # FIX: Use indices [0:3] for translation, not [4]
# FIX: Use indices [0:3] for translation, not [4]
    t_next_vec = np.array([float(v) for v in gt_next[0:3]])
    t_cur_vec  = np.array([float(v) for v in gt_cur[0:3]])

    R_gt_rel = R_gt_next @ R_gt_cur.T
    # Correct formula: t_rel = t_next - R_rel @ t_cur
    t_gt_rel = (t_next_vec - R_gt_rel @ t_cur_vec).reshape(3, 1)

    R_gt_rel = R_gt_next @ R_gt_cur.T
    # Correct for
    # ula: t_rel = t_next - R_rel @ t_cur
    t_gt_rel = (t_next_vec - R_gt_rel @ t_cur_vec).reshape(3, 1)
    # --- END OF FIX ---
    

    

