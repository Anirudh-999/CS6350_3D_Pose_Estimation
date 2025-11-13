import cv2
import numpy as np
from custom_logging import info, warn, show_and_save
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R_transform

# --- NEW HELPER FUNCTION (Moved from data_handling.py) ---
def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (in degrees).
    Returns angles as [roll, pitch, yaw] in ZYX order (extrinsic).
    """
    if R is None:
        return np.array([0.0, 0.0, 0.0])
        
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

# --- KEPT FROM YOUR ORIGINAL ---
def estimate_fundamental(pts1, pts2):
    if pts1.shape[0] < 8:
        warn("Need at least 8 points to estimate Fundamental matrix reliably.")
        return None, None
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99) # Tighter threshold
    return F, mask

# --- KEPT FROM YOUR ORIGINAL ---
def draw_epipolar_lines(img1, img2, pts1, pts2, F, fname):
    if F is None or pts1 is None or pts2 is None or pts1.shape[0] == 0 or pts2.shape[0] == 0:
        warn("No fundamental matrix or point correspondences to draw epipolar lines.")
        return
    # (The rest of this function is fine as-is)
    pts1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)

    def draw_lines(img, lines, pts):
        out = img.copy()
        h, w = out.shape[:2]
        for r, p in zip(lines, pts):
            a, b, c = r
            if abs(b) < 1e-6:
                y0, y1 = 0, h
                x0 = int(-c / a) if abs(a) > 1e-6 else 0
                x1 = x0
            else:
                x0, y0 = 0, int(-c / b)
                x1, y1 = w, int((-c - a * w) / b)
            cv2.line(out, (int(x0), int(np.clip(y0, 0, h - 1))),
                          (int(x1), int(np.clip(y1, 0, h - 1))), (0, 255, 0), 1)
            cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
        return out

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    img1_lines = draw_lines(img1, lines1, pts1)
    img2_lines = draw_lines(img2, lines2, pts2)
    h1, h2 = img1_lines.shape[0], img2_lines.shape[0]
    max_h = max(h1, h2)
    if h1 < max_h:
        img1_lines = cv2.copyMakeBorder(img1_lines, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if h2 < max_h:
        img2_lines = cv2.copyMakeBorder(img2_lines, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    vis = np.hstack([img1_lines, img2_lines])
    show_and_save(vis, "Epipolar lines L|R", fname)


# --- THIS IS THE MAIN NEW FUNCTION ---
# It replaces your old estimate_essential_and_pose
def self_calibrate_and_find_pose(pts1, pts2, K_guess, img_shape):
    """
    Robustly finds K, R, and t from "near-planar" features.
    It separates planar (H) from non-planar (F) points to
    get a stable pose.
    """
    h, w = img_shape[:2]
    
    if K_guess is None:
        # Create a default guess if none is provided
        f_guess = max(h, w) # A common heuristic
        K_guess = np.array([[f_guess, 0, w/2], [0, f_guess, h/2], [0, 0, 1]])
        warn(f"No K provided, guessing focal length={f_guess}")
    else:
        info(f"Using provided K/calib as initial guess.")

    # --- 1. Find the dominant plane (H) ---
    H, h_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0, 0.99)
    if H is None:
        warn("H matrix estimation failed. Aborting.")
        return None, None, K_guess, None # Return guess
        
    h_mask = h_mask.ravel().astype(bool)
    
    # --- 2. Find non-planar points ---
    # These are the "outliers" to the dominant plane
    non_planar_mask = ~h_mask
    non_planar_pts1 = pts1[non_planar_mask]
    non_planar_pts2 = pts2[non_planar_mask]

    info(f"Total matches: {len(pts1)}. Planar (H) inliers: {np.sum(h_mask)}. Non-planar (3D) outliers: {len(non_planar_pts1)}")

    if len(non_planar_pts1) < 8:
        warn("Not enough non-planar points (<8). Pose will be unstable.")
        return None, None, K_guess, non_planar_mask

    # --- 3. Get a stable F from ONLY non-planar points ---
    F_clean, f_mask = cv2.findFundamentalMat(non_planar_pts1, non_planar_pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F_clean is None:
        warn("Clean F matrix estimation failed. Aborting.")
        return None, None, K_guess, non_planar_mask

    # --- 4. Optimize focal length (f) using the clean F ---
    def cost_function(f_scalar):
        f = float(np.atleast_1d(f_scalar)[0])
        K_temp = K_guess.copy()
        K_temp[0, 0] = f
        K_temp[1, 1] = f
        E_temp = K_temp.T @ F_clean @ K_temp
        U, S, Vt = np.linalg.svd(E_temp)
        # Minimize difference of two largest singular values
        return (S[0] - S[1]) ** 2 

    f_init = K_guess[0, 0]
    result = minimize(cost_function, [f_init], bounds=[(100, 4000)], method='L-BFGS-B')
    f_optimized = float(result.x[0])
    
    K_optimized = K_guess.copy()
    K_optimized[0, 0] = f_optimized
    K_optimized[1, 1] = f_optimized
    info(f"Optimized focal length: {f_optimized:.2f}")

    # --- 5. Recover Pose with optimized K ---
    E_final = K_optimized.T @ F_clean @ K_optimized
    
    # Get inliers from the F_clean calculation
    clean_pts1 = non_planar_pts1[f_mask.ravel().astype(bool)]
    clean_pts2 = non_planar_pts2[f_mask.ravel().astype(bool)]

    if len(clean_pts1) < 5:
        warn("Too few F-inliers after cleaning. Aborting pose recovery.")
        return None, None, K_optimized, non_planar_mask

    # Use the same non-planar points and the new E to get the pose
    _, R_est, t_est, mask_pose = cv2.recoverPose(E_final, clean_pts1, clean_pts2, K_optimized)
    
    info(f"Recovered pose with {np.sum(mask_pose)} inliers (from {len(clean_pts1)} non-planar pts)")
    
    return R_est, t_est, K_optimized, non_planar_mask


# --- KEPT FROM YOUR ORIGINAL ---
def rotation_angle_error_deg(R_pred, R_gt):
    if R_pred is None or R_gt is None: return None
    R_err = R_gt.T @ R_pred
    rot = R_transform.from_matrix(R_err)
    return float(np.degrees(rot.magnitude()))

# --- KEPT FROM YOUR ORIGINAL ---
def translation_direction_error_deg(t_pred, t_gt):
    if t_pred is None or t_gt is None: return None
    tp = np.asarray(t_pred).ravel().astype(float)
    tg = np.asarray(t_gt).ravel().astype(float)
    if np.linalg.norm(tp) < 1e-9 or np.linalg.norm(tg) < 1e-9:
        return None
    tp /= np.linalg.norm(tp)
    tg /= np.linalg.norm(tg)
    dot = np.clip(np.dot(tp, tg), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))

# --- KEPT FROM YOUR ORIGINAL ---
def reprojection_rms(pts1, pts2, Rmat, tvec, K):
    if pts1 is None or pts2 is None or Rmat is None or tvec is None or K is None or len(pts1) < 1:
        return None
    pts1 = np.asarray(pts1, dtype=float)
    pts2 = np.asarray(pts2, dtype=float)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K @ np.hstack([Rmat, tvec.reshape(3,1)])
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    proj2 = (P2 @ np.hstack([pts3d, np.ones((pts3d.shape[0],1))]).T).T
    proj2 = proj2[:, :2] / proj2[:, 2:3]
    diff = pts2 - proj2
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

# --- MODIFIED FROM YOUR ORIGINAL (Added translation magnitude error) ---
def evaluate_pose_accuracy(R_pred, t_pred, R_gt, t_gt, pts1=None, pts2=None, K=None):
    """Convenience wrapper that prints rotation, translation, and RMS errors."""
    r_err = rotation_angle_error_deg(R_pred, R_gt)
    t_dir_err = translation_direction_error_deg(t_pred, t_gt)
    
    # Calculate magnitude error
    t_mag_err = None
    if t_pred is not None and t_gt is not None:
        t_mag_pred = np.linalg.norm(t_pred)
        t_mag_gt = np.linalg.norm(t_gt)
        if t_mag_gt > 1e-6: # Avoid division by zero
            t_mag_err = (np.abs(t_mag_pred - t_mag_gt) / t_mag_gt) * 100
    
    rms = reprojection_rms(pts1, pts2, R_pred, t_pred, K) if pts1 is not None else None
    
    print("---- Pose Accuracy ----")
    print(f"Rotation error: {r_err:.3f}°" if r_err is not None else "Rotation error: N/A")
    print(f"Translation dir. error: {t_dir_err:.3f}°" if t_dir_err is not None else "Translation error: N/A")
    print(f"Translation scale error: {t_mag_err:.2f}%" if t_mag_err is not None else "Translation scale error: N/A")
    if rms is not None:
        print(f"Reprojection RMS: {rms:.3f} px")
    print("------------------------")
    return {"rot_err_deg": r_err, "trans_dir_err_deg": t_dir_err, "trans_scale_err_pct": t_mag_err, "rms_px": rms}