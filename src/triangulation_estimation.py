from custom_logging import info, warn
import numpy as np
import cv2
from scipy.optimize import least_squares

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except Exception:
    o3d = None
    _O3D_AVAILABLE = False

def triangulate_with_P(pts1, pts2, P1, P2):
    """
    pts1, pts2: Nx2 float arrays. P1, P2: 3x4 projection matrices.
    Returns Nx3 points.
    """
    if pts1.shape[0] == 0:
        return np.empty((0,3))
    pts1_h = pts1.T
    pts2_h = pts2.T
    pts4 = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    pts3 = (pts4[:3] / pts4[3]).T
    return pts3

def pose_from_depth_icp(depthL, depthR, fx=700, fy=700, cx=None, cy=None, baseline=0.1):
    """
    Estimate relative pose (R, t) between left/right using dense depth maps.
    Fallback if Essential matrix fails or intrinsics unknown.
    """
    if not _O3D_AVAILABLE:
        print("[WARN] Open3D not available — skipping ICP pose refinement.")
        return None, None

    h, w = depthL.shape
    if cx is None: cx = w / 2
    if cy is None: cy = h / 2

    # Reconstruct point clouds from disparity
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    ZL = fx * baseline / (depthL + 1e-6)
    XL = (xx - cx) * ZL / fx
    YL = (yy - cy) * ZL / fy

    ZR = fx * baseline / (depthR + 1e-6)
    XR = (xx - cx) * ZR / fx
    YR = (yy - cy) * ZR / fy

    ptsL = np.stack((XL, YL, ZL), axis=-1).reshape(-1, 3)
    ptsR = np.stack((XR, YR, ZR), axis=-1).reshape(-1, 3)

    # Filter valid points
    mask = np.isfinite(ptsL).all(axis=1) & np.isfinite(ptsR).all(axis=1)
    ptsL, ptsR = ptsL[mask], ptsR[mask]

    # Convert to Open3D clouds
    pcdL = o3d.geometry.PointCloud()
    pcdL.points = o3d.utility.Vector3dVector(ptsL)
    pcdR = o3d.geometry.PointCloud()
    pcdR.points = o3d.utility.Vector3dVector(ptsR)

    # ICP registration
    threshold = 0.05
    trans_init = np.eye(4)
    reg = o3d.pipelines.registration.registration_icp(
        pcdL, pcdR, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    T = reg.transformation
    R = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)
    print("[INFO] ICP-based rotation:\n", R)
    print("[INFO] ICP-based translation:\n", t.ravel())
    return R, t

def icp_refine(observed_pts, model_pts, init=np.eye(4)):
    if not _O3D_AVAILABLE:
        warn("Open3D not available; skipping ICP.")
        return init
    if observed_pts.shape[0] < 20 or model_pts is None or model_pts.shape[0] < 20:
        warn("Insufficient points for ICP.")
        return init
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_pts))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed_pts))
    src_down = src.voxel_down_sample(0.01)
    tgt_down = tgt.voxel_down_sample(0.01)
    src_down.estimate_normals()
    tgt_down.estimate_normals()
    thresh = 0.05
    reg = o3d.pipelines.registration.registration_icp(src_down, tgt_down, thresh, init,
                                                     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    info(f"ICP done: fitness={reg.fitness:.4f}, rmse={reg.inlier_rmse:.4f}")
    return reg.transformation

def bundle_adjustment(pts1, pts2, K, R, t):
    """
    Refine R, t by minimizing geometric reprojection error.
    """
    def reprojection_error(params, pts1, pts2, K):
        R_vec = params[:3]
        t = params[3:]
        R, _ = cv2.Rodrigues(R_vec)
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t.reshape(3, 1)))
        pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T
        pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3).T
        proj1 = P1 @ pts1_h
        proj2 = P2 @ pts2_h
        proj1 /= proj1[2]
        proj2 /= proj2[2]
        error = np.linalg.norm(proj1[:2] - proj2[:2], axis=0)
        return error

    # Initial parameters
    R_vec, _ = cv2.Rodrigues(R)
    params_init = np.hstack((R_vec.ravel(), t.ravel()))

    # Optimize
    result = least_squares(reprojection_error, params_init, args=(pts1, pts2, K))
    R_refined, _ = cv2.Rodrigues(result.x[:3])
    t_refined = result.x[3:]

    return R_refined, t_refined

def refine_pose_with_bundle_adjustment(pts1, pts2, K, R, t):
    """
    Wrapper for bundle adjustment to refine pose parameters.
    Args:
        pts1: Nx2 array of points in image 1.
        pts2: Nx2 array of points in image 2.
        K: Camera intrinsic matrix.
        R: Initial rotation matrix.
        t: Initial translation vector.
    Returns:
        Refined rotation matrix and translation vector.
    """
    info("Starting bundle adjustment to refine pose parameters.")
    R_refined, t_refined = bundle_adjustment(pts1, pts2, K, R, t)
    info("Bundle adjustment completed.")
    info(f"Refined Rotation Matrix:\n{R_refined}")
    info(f"Refined Translation Vector:\n{t_refined}")
    return R_refined, t_refined