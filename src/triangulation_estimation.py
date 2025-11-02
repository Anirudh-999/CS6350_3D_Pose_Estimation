
from custom_logging import info, warn
import numpy as np
import cv2

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