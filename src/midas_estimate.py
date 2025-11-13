import numpy as np
import cv2
import time
import os
import sys
from scipy.optimize import minimize

# --- Imports from your project files ---
from custom_logging import info, warn, err
from input_output import load_image_rgb
from feature_extraction_maping import extract_features_sift, match_descriptors_flann
from epipolar_geometry import estimate_fundamental # For K estimation
from disparity_estimation import estimate_depth_midas, _MIDAS_AVAILABLE
from triangulation_estimation import icp_refine, _O3D_AVAILABLE

def estimate_focal_length(pts1, pts2, w=640, h=480):
    """
    Estimates focal length by optimizing the Essential Matrix singular values.
    This logic is borrowed from your epipolar_geometry.py
    """
    if pts1.shape[0] < 8:
        warn("Need at least 8 points for focal length estimation.")
        return None
        
    F, maskF = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    if F is None or F.shape != (3, 3):
        warn("Fundamental matrix estimation failed.")
        return None

    # Initial K guess
    cx, cy = w / 2.0, h / 2.0
    f_init = (w + h) / 2.0
    
    def cost_function(f_scalar):
        f = float(np.atleast_1d(f_scalar)[0])
        K_temp = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        E_temp = K_temp.T @ F @ K_temp
        U, S, Vt = np.linalg.svd(E_temp)
        # Cost is deviation from [s, s, 0] singular value property
        return (S[0] - S[1]) ** 2 + (S[2]) ** 2

    result = minimize(cost_function, [f_init], bounds=[(100, 2000)], method='L-BFGS-B')
    f_optimized = float(result.x[0])
    
    K_est = np.array([
        [f_optimized, 0, cx],
        [0, f_optimized, cy],
        [0, 0, 1]
    ])
    info(f"Optimized focal length: {f_optimized:.2f}")
    return K_est

def create_point_cloud(depth_map, K, step=5):
    """
    Unprojects a depth map into a 3D point cloud given intrinsics K.
    Downsamples by 'step' for efficiency.
    """
    if depth_map is None or K is None:
        return None
        
    h, w = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create a grid of (u, v) coordinates
    v, u = np.indices((h, w))
    
    # Downsample
    v, u = v[::step, ::step], u[::step, ::step]
    Z = depth_map[::step, ::step]
    
    # Unproject
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    pts3d = np.stack((X, Y, Z), axis=-1)
    
    # Filter out invalid points (e.g., if depth was 0)
    pts3d = pts3d.reshape(-1, 3)
    pts3d = pts3d[np.all(np.isfinite(pts3d), axis=1)]
    
    return pts3d

def normalize_point_cloud(pts3d):
    """
    Aligns point cloud to origin and scales to unit norm.
    This helps correct for the unknown/inconsistent scale from MiDaS.
    """
    if pts3d is None or pts3d.shape[0] == 0:
        return None
        
    # Center the cloud
    centroid = np.mean(pts3d, axis=0)
    pts3d_centered = pts3d - centroid
    
    # Scale to unit norm
    norm = np.linalg.norm(pts3d_centered)
    if norm < 1e-6:
        return pts3d_centered # Avoid division by zero
        
    pts3d_normalized = pts3d_centered / norm
    return pts3d_normalized


def estimate_pose_with_midas_icp(L_rgb, R_rgb):
    """
    Main function to estimate pose (R, t) using MiDaS and ICP.
    """
    if not _MIDAS_AVAILABLE or not _O3D_AVAILABLE:
        warn("MiDaS or Open3D not available. Skipping ICP pipeline.")
        return None, None
        
    h, w = L_rgb.shape[:2]
    
    # --- Step 1: Estimate Focal Length (Intrinsics K) ---
    info("Estimating focal length from 2D matches...")
    kpsL, descL = extract_features_sift(L_rgb)
    kpsR, descR = extract_features_sift(R_rgb)
    matches = match_descriptors_flann(descL, descR)
    
    if len(matches) < 8:
        warn("Not enough SIFT matches to estimate K. Using heuristic.")
        f_est = (w + h) / 2.0
        K_est = np.array([[f_est, 0, w/2.0], [0, f_est, h/2.0], [0, 0, 1]])
    else:
        ptsL = np.array([kpsL[m.queryIdx].pt for m in matches])
        ptsR = np.array([kpsR[m.trainIdx].pt for m in matches])
        K_est = estimate_focal_length(ptsL, ptsR, w, h)
        if K_est is None:
            warn("Focal length estimation failed. Using heuristic.")
            f_est = (w + h) / 2.0
            K_est = np.array([[f_est, 0, w/2.0], [0, f_est, h/2.0], [0, 0, 1]])

    # --- Step 2: Get MiDaS Depth Maps ---
    info("Running MiDaS depth estimation for Left image...")
    depthL = estimate_depth_midas(L_rgb)
    info("Running MiDaS depth estimation for Right image...")
    depthR = estimate_depth_midas(R_rgb)
    
    if depthL is None or depthR is None:
        err("Failed to get MiDaS depth maps.")
        return None, None
        
    # --- Step 3: Create 3D Point Clouds ---
    info("Creating point clouds from depth maps...")
    pts3d_L_raw = create_point_cloud(depthL, K_est)
    pts3d_R_raw = create_point_cloud(depthR, K_est)

    if pts3d_L_raw is None or pts3d_R_raw is None:
        err("Failed to create point clouds.")
        return None, None

    info(f"Raw Point Clouds: L={pts3d_L_raw.shape[0]} pts, R={pts3d_R_raw.shape[0]} pts")

    # --- Step 4: Normalize Point Clouds (to handle scale) ---
    info("Normalizing point clouds to align scale...")
    pts3d_L_norm = normalize_point_cloud(pts3d_L_raw)
    pts3d_R_norm = normalize_point_cloud(pts3d_R_raw)
    
    if pts3d_L_norm is None or pts3d_R_norm is None:
        err("Failed to normalize point clouds.")
        return None, None

    # --- Step 5: Align with ICP ---
    info("Running ICP to align point clouds...")
    # Find transform that maps R -> L (i.e., L = T @ R)
    # We use R as source (model) and L as target (observed)
    T_matrix = icp_refine(observed_pts=pts3d_L_norm, model_pts=pts3d_R_norm)
    
    R_est = T_matrix[:3, :3]
    t_est = T_matrix[:3, 3] # Note: This 't' is relative to the normalized clouds
    
    return R_est, t_est

# --- Main execution block to run this file directly ---
if __name__ == "__main__":
    
    # --- Define Image Paths (Update these as needed) ---
    # Using the paths from data_handling.py as a default
    try:
        from data_handling import LEFT_IMG_PATH, RIGHT_IMG_PATH
        info(f"Using image paths from data_handling.py")
    except Exception:
        warn("data_handling.py not found. Using hardcoded paths.")
        # Fallback paths
        LEFT_IMG_PATH  = "../images/img3.jpeg"
        RIGHT_IMG_PATH = "../images/img4.jpeg"

    if not os.path.exists(LEFT_IMG_PATH) or not os.path.exists(RIGHT_IMG_PATH):
        err(f"Images not found at: {LEFT_IMG_PATH} or {RIGHT_IMG_PATH}")
        sys.exit(1)

    info("Loading images...")
    L_rgb = load_image_rgb(LEFT_IMG_PATH)
    R_rgb = load_image_rgb(RIGHT_IMG_PATH)
    
    t0 = time.time()
    info("Starting MiDaS + ICP Pose Estimation Pipeline...")
    
    R, t = estimate_pose_with_midas_icp(L_rgb, R_rgb)
    
    if R is not None:
        info("--- MiDaS + ICP Pipeline Succeeded ---")
        info(f"Estimated Rotation (R):\n{R}")
        info(f"Estimated (Normalized) Translation (t):\n{t.T}")
    else:
        err("--- MiDaS + ICP Pipeline Failed ---")
        
    info(f"Total runtime: {time.time() - t0:.2f}s")