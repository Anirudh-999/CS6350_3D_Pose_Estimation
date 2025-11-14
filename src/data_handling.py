# data_handling.py
import os
import numpy as np
from typing import Dict
from main import run_pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------
# Paths / Setup
# ---------------------------------------------------------
path = os.path.dirname(os.path.abspath(__file__))
rgb_files_path = os.path.join(path, "..", "dataset", "rgb.txt")
ground_path    = os.path.join(path, "..", "dataset", "groundtruth.txt")

YOLO_WEIGHTS = "yolov8l-seg.pt"
CALIB_PATH   = None
OUTPUT_DIR   = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Quaternion → Rotation Matrix
# ---------------------------------------------------------
def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2*q2 + q3*q3),   2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),       1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),       2*(q2*q3 + q0*q1),     1 - 2*(q1*q1 + q2*q2)]
    ])

# ---------------------------------------------------------
# Rotation Matrix → Euler angles (deg, ZYX)
# ---------------------------------------------------------
def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0
    return np.degrees([roll, pitch, yaw])


# ---------------------------------------------------------
# Load dataset + GT
# ---------------------------------------------------------
rgb_paths: Dict[str, Dict[str, list]] = {}

with open(rgb_files_path, "r") as file:
    for line in file:
        t_str, img_path = line.strip().split()
        ts = round(float(t_str), 2)
        key = f"{ts:.2f}"
        rgb_paths[key] = {
            "image": os.path.join("../dataset", img_path),
            "pose": None
        }

with open(ground_path, "r") as gr:
    for line in gr:
        t_str, tx, ty, tz, qx, qy, qz, qw = line.strip().split()
        ts = round(float(t_str), 2)
        key = f"{ts:.2f}"
        if key in rgb_paths:
            R = quaternion_to_rotation_matrix([float(qw), float(qx), float(qy), float(qz)])
            euler = rotation_matrix_to_euler_angles(R)
            rgb_paths[key]["pose"] = [float(tx), float(ty), float(tz), R, euler]

# keep only keys having pose
rgb_paths = {k: v for k,v in rgb_paths.items() if v["pose"] is not None}
keys = sorted(rgb_paths.keys())


# ---------------------------------------------------------
# Lists for GT & EST angles
# ---------------------------------------------------------
roll_gt_list = []
pitch_gt_list = []
yaw_gt_list = []

roll_est_list = []
pitch_est_list = []
yaw_est_list = []


# ---------------------------------------------------------
# Loop through pairs
# ---------------------------------------------------------
num_pairs = min(len(keys)-1, 5000)

for i in tqdm(range(num_pairs), desc="Processing image pairs"):

    cur_key  = keys[i]
    next_key = keys[i+1]

    cur  = rgb_paths[cur_key]
    nxt  = rgb_paths[next_key]

    # --- RUN PIPELINE ---
    results, metrics, per_frame = run_pipeline(
        left_path  = nxt["image"],
        right_path = cur["image"],
        calib_path = CALIB_PATH,
        yolo_weights = YOLO_WEIGHTS,
        gt_pose_cur  = cur["pose"],
        gt_pose_next = nxt["pose"]
    )

    # --- STORE GT EULER ANGLES ---
    roll_gt, pitch_gt, yaw_gt = nxt["pose"][4]
    R_cur = cur["pose"][3]
    R_next = nxt["pose"][3]
    R_gt_rel = R_next @ R_cur.T
    
    # this is the true GT for comparison
    euler_gt_rel = rotation_matrix_to_euler_angles(R_gt_rel)
    
    roll_gt_list.append(euler_gt_rel[0])
    pitch_gt_list.append(euler_gt_rel[1])
    yaw_gt_list.append(euler_gt_rel[2])
    

    # --- STORE ESTIMATED EULER ANGLES ---
    if results is not None and len(results) > 0:
        R_est = results[0]["R"]
        e_est = rotation_matrix_to_euler_angles(R_est)
        roll_est_list.append(e_est[0])
        pitch_est_list.append(e_est[1])
        yaw_est_list.append(e_est[2])
    else:
        roll_est_list.append(np.nan)
        pitch_est_list.append(np.nan)
        yaw_est_list.append(np.nan)



# ---------------------------------------------------------
# Plot GT vs EST for each angle
# ---------------------------------------------------------
def plot_gt_vs_est(gt_vals, est_vals, title, ylabel, filename):
    x = np.arange(len(gt_vals))

    plt.figure(figsize=(12, 4))
    plt.plot(x, gt_vals, label="Ground Truth", linewidth=2)
    plt.plot(x, est_vals, label="Estimated",   linewidth=2)
    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"[PLOT SAVED] {filename}")


plot_gt_vs_est(roll_gt_list,  roll_est_list,
               "Roll Angle: GT vs Estimated", "Roll (deg)",
               "roll_gt_vs_est.png")

plot_gt_vs_est(pitch_gt_list, pitch_est_list,
               "Pitch Angle: GT vs Estimated", "Pitch (deg)",
               "pitch_gt_vs_est.png")

plot_gt_vs_est(yaw_gt_list,   yaw_est_list,
               "Yaw Angle: GT vs Estimated", "Yaw (deg)",
               "yaw_gt_vs_est.png")


print("\nDone. Plots saved in:", OUTPUT_DIR)
