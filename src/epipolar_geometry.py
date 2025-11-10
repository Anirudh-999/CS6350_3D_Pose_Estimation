import cv2
import numpy as np
from custom_logging import info, warn, show_and_save
from scipy.optimize import minimize

def estimate_fundamental(pts1, pts2):
    if pts1.shape[0] < 8:
        warn("Need at least 8 points to estimate Fundamental matrix reliably.")
        return None, None
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    return F, mask

def draw_epipolar_lines(img1, img2, pts1, pts2, F, fname):
    # draw epipolar lines on both images for the first few inliers
    if F is None or pts1 is None or pts2 is None or pts1.shape[0] == 0 or pts2.shape[0] == 0:
        warn("No fundamental matrix or point correspondences to draw epipolar lines.")
        return

    # ensure correct dtype/shape
    pts1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float32).reshape(-1, 2)

    def draw_lines(img, lines, pts):
        out = img.copy()
        h, w = out.shape[:2]
        for r, p in zip(lines, pts):
            a, b, c = r
            # avoid division by zero for horizontal lines (b==0)
            if abs(b) < 1e-6:
                y0 = 0
                y1 = h
                x0 = int(-c / a) if abs(a) > 1e-6 else 0
                x1 = x0
            else:
                x0, y0 = 0, int(-c / b)
                x1, y1 = w, int((-c - a * w) / b)
            cv2.line(out, (int(x0), int(np.clip(y0, 0, h - 1))),
                          (int(x1), int(np.clip(y1, 0, h - 1))), (0, 255, 0), 1)
            cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
        return out

    # compute epipolar lines
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    img1_lines = draw_lines(img1, lines1, pts1)
    img2_lines = draw_lines(img2, lines2, pts2)

    # Pad images to the same height before horizontal stacking to avoid dimension mismatch
    h1, h2 = img1_lines.shape[0], img2_lines.shape[0]
    max_h = max(h1, h2)
    if h1 < max_h:
        img1_lines = cv2.copyMakeBorder(img1_lines, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if h2 < max_h:
        img2_lines = cv2.copyMakeBorder(img2_lines, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    vis = np.hstack([img1_lines, img2_lines])
    show_and_save(vis, "Epipolar lines L|R", fname)

# ---------- Essential & pose ----------
def estimate_essential_and_pose(pts1, pts2, K=None):
    if pts1.shape[0] < 5:
        warn("Too few points to estimate Essential matrix.")
        return None, None, None

    if K is None:
        # Heuristic K (f ~ width)
        w = int(np.max(pts1[:, 0]) * 2) if pts1.size else 640
        fx = fy = float(w)
        cx = cy = w / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        warn("No intrinsics K: approximating; pose will be scale-ambiguous.")

    # Estimate F once outside optimization
    F, maskF = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    if F is None or F.shape != (3, 3):
        warn("Fundamental matrix estimation failed — cannot optimize focal length.")
        return None, None, None

    # Optimize focal length f
    def cost_function(f_scalar):
        f = float(np.atleast_1d(f_scalar)[0])
        K_temp = np.array([
            [f, 0, K[0, 2]],
            [0, f, K[1, 2]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Build temporary essential matrix
        E_temp = K_temp.T @ F @ K_temp

        # SVD structure constraint: two equal singular values, one zero
        U, S, Vt = np.linalg.svd(E_temp)
        # Use squared difference to ensure smooth cost
        return (S[0] - S[1]) ** 2 + (S[2]) ** 2

    result = minimize(cost_function, [K[0, 0]], bounds=[(100, 2000)], method='L-BFGS-B')
    f_optimized = float(result.x[0])
    K[0, 0] = K[1, 1] = f_optimized

    # Compute essential matrix with refined K
    E, maskE = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    if E is None:
        warn("Essential estimation failed.")
        return None, None, None

    # Recover pose
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    info(f"Recovered pose with {np.sum(mask_pose)} inliers (of {len(pts1)})")
    info(f"Optimized focal length: {f_optimized:.2f}")

    return R, t, mask_pose


import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

def rotation_angle_error_deg(R_pred, R_gt):
    """Total angular difference in degrees between two rotation matrices."""
    if R_pred is None or R_gt is None:
        return None
    R_err = R_gt.T @ R_pred
    rot = R.from_matrix(R_err)
    return float(np.degrees(rot.magnitude()))

def rotation_euler_error_deg(R_pred, R_gt, seq='xyz', degrees=True):
    """Return per-axis Euler rotation error."""
    if R_pred is None or R_gt is None:
        return None
    R_err = R_gt.T @ R_pred
    return R.from_matrix(R_err).as_euler(seq, degrees=degrees)

def translation_direction_error_deg(t_pred, t_gt):
    """Angle between translation directions (scale ambiguous) in degrees."""
    if t_pred is None or t_gt is None:
        return None
    tp = np.asarray(t_pred).ravel().astype(float)
    tg = np.asarray(t_gt).ravel().astype(float)
    if np.linalg.norm(tp) < 1e-9 or np.linalg.norm(tg) < 1e-9:
        return None
    tp /= np.linalg.norm(tp)
    tg /= np.linalg.norm(tg)
    dot = np.clip(np.dot(tp, tg), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))

def reprojection_rms(pts1, pts2, Rmat, tvec, K):
    """Compute reprojection RMS error (in pixels)."""
    if pts1 is None or pts2 is None or Rmat is None or tvec is None or K is None:
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

def evaluate_pose_accuracy(R_pred, t_pred, R_gt, t_gt, pts1=None, pts2=None, K=None):
    print("hERE")
    """Convenience wrapper that prints rotation, translation, and RMS errors."""
    r_err = rotation_angle_error_deg(R_pred, R_gt)
    t_err = translation_direction_error_deg(t_pred, t_gt)
    rms = reprojection_rms(pts1, pts2, R_pred, t_pred, K) if pts1 is not None else None
    print("---- Pose Accuracy ----")
    print(f"Rotation error: {r_err:.3f}°" if r_err is not None else "Rotation error: N/A")
    print(f"Translation dir. error: {t_err:.3f}°" if t_err is not None else "Translation error: N/A")
    if rms is not None:
        print(f"Reprojection RMS: {rms:.3f} px")
    print("------------------------")
    return {"rot_err_deg": r_err, "trans_dir_err_deg": t_err, "rms_px": rms}
