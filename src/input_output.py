import cv2
import os
import numpy as np
from custom_logging import info, warn, show_and_save
from typing import Dict
import math

# LEFT_IMG_PATH  = "images/img2.png"      # path to left image
# RIGHT_IMG_PATH = "images/img1.png"      # path to right image

LEFT_IMG_PATH  = "../images/img3.jpeg"      # path to left image
RIGHT_IMG_PATH = "../images/img4.jpeg"      # path to right image

CALIB_PATH     = "calib.txt"      # optional KITTI-style calib (P0/P1 or K)
YOLO_WEIGHTS   = "yolov8l.pt"     # optional (if ultralytics installed)
OUTPUT_DIR     = "./output"       # where we save visualizations
USE_GPU        = True             # preference flag

def load_image_rgb(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError(f"cv2 failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #show_and_save(rgb, title=os.path.basename(path), fname=os.path.basename(path), output_dir=OUTPUT_DIR)
    return rgb

def load_calibration_kitti_like(path: str) -> Dict[str, np.ndarray]:
    """
    Parse a KITTI-style text file with P0:, P1:, ... or K, K0, K1 lines.
    Returns dict possibly containing 'P0','P1','K0','K1','baseline'.
    """
    if not os.path.exists(path):
        warn("Calibration file not found, proceeding without calibration.")
        return {}
    calib = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, val = line.split(':',1)
            nums = [float(x) for x in val.strip().split()]
            key = key.strip()
            if key.startswith('P') and len(nums) == 12:
                calib[key] = np.array(nums).reshape(3,4)
            elif key in ('K','K0','K1') and len(nums) == 4:
                fx, fy, cx, cy = nums
                calib[key] = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
            elif key in ('K','K0','K1') and len(nums) == 9:
                calib[key] = np.array(nums).reshape(3,3)
    # derive K0/K1 from P0/P1 if present
    if 'P0' in calib and 'P1' in calib:
        calib.setdefault('K0', calib['P0'][:, :3].copy())
        calib.setdefault('K1', calib['P1'][:, :3].copy())
        t0 = calib['P0'][:,3]; t1 = calib['P1'][:,3]
        calib['baseline'] = float(np.linalg.norm(t1 - t0))
    info(f"Calibration keys: {list(calib.keys())}")
    return calib

def yaw_from_R(R, R_ref=None):
    """
    Return yaw angle(s) in degrees from rotation matrix R.
    If R_ref is provided, calculate relative yaw with respect to R_ref.
    """
    try:
        if R_ref is not None:
            # Compute relative rotation matrix
            R_rel = R_ref.T @ R
        else:
            R_rel = R

        # Calculate yaw using two conventions
        yaw_a = math.degrees(math.atan2(R_rel[1, 0], R_rel[0, 0]))
        yaw_b = math.degrees(math.atan2(R_rel[0, 2], R_rel[2, 2]))
        return {'yaw_a_deg': yaw_a, 'yaw_b_deg': yaw_b}
    except Exception:
        return {}

def yaw_from_pointcloud_pca(pts3):
    if pts3.shape[0] < 3:
        return None
    pts = pts3 - pts3.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eig(cov)
    idx = int(np.argmax(vals))
    principal = vecs[:, idx]
    yaw = math.degrees(math.atan2(principal[1], principal[0]))
    return yaw

def visualize_keypoints(img_rgb, kps, fname="keypoints.png", title="Keypoints"):
    img = img_rgb.copy()
    kps_cv = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), _size=3) if not hasattr(p,'pt') else p for p in kps]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = cv2.drawKeypoints(img_bgr, kps_cv, None, color=(0,255,0))
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    show_and_save(out_rgb, title, fname)