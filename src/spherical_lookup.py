
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rrot

def estimate_rotation_spherical(img_src_bgr, img_tgt_bgr, f_init=None, pyramid_levels=3, px_sample_step=2):
    """
    Estimate rotation R mapping src -> tgt using photometric alignment on a sphere.
    Assumes pure rotation (no translation).
    Returns (R_est, f_est)
    """
    src_gray = cv2.cvtColor(img_src_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    tgt_gray = cv2.cvtColor(img_tgt_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = src_gray.shape

    def K_from_f(f, w, h):
        cx, cy = w/2.0, h/2.0
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    def compute_bearing_map(w, h, K):
        fx = K[0, 0]; cx = K[0, 2]; cy = K[1, 2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x = (u - cx) / fx
        y = (v - cy) / fx
        z = np.ones_like(x)
        vec = np.stack([x, y, z], axis=2)
        vec /= np.linalg.norm(vec, axis=2, keepdims=True)
        return vec

    def warp_source(src, bearings_tgt, Rmat, K):
        h, w = bearings_tgt.shape[:2]
        dirs_src = (Rmat.T @ bearings_tgt.reshape(-1,3).T).T
        u = (dirs_src[:,0]/dirs_src[:,2])*K[0,0] + K[0,2]
        v = (dirs_src[:,1]/dirs_src[:,2])*K[1,1] + K[1,2]
        mapx = u.reshape(h,w).astype(np.float32)
        mapy = v.reshape(h,w).astype(np.float32)
        warped = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return warped

    # pyramid
    pyr_src = [src_gray]
    pyr_tgt = [tgt_gray]
    for _ in range(1, pyramid_levels):
        pyr_src.insert(0, cv2.pyrDown(pyr_src[0]))
        pyr_tgt.insert(0, cv2.pyrDown(pyr_tgt[0]))

    rvec = np.zeros(3)
    f = f_init or max(w, h)

    for lvl in range(pyramid_levels):
        src = pyr_src[lvl]
        tgt = pyr_tgt[lvl]
        hL, wL = tgt.shape
        s = wL / w
        fL = f * s
        K = K_from_f(fL, wL, hL)
        bearings = compute_bearing_map(wL, hL, K)
        bearings = bearings[::px_sample_step, ::px_sample_step]
        src = src[::px_sample_step, ::px_sample_step]
        tgt = tgt[::px_sample_step, ::px_sample_step]

        def residuals(rvec_local):
            Rmat = Rrot.from_rotvec(rvec_local).as_matrix()
            warped = warp_source(src, bearings, Rmat, K)

            # fixed-length residuals (no masking)
            diff = warped - tgt
            # remove NaNs and inf (can appear near image borders)
            diff[np.isnan(diff)] = 0
            diff[np.isinf(diff)] = 0
            return diff.ravel()


        res = least_squares(residuals, rvec, method='trf', loss='huber', f_scale=0.1, max_nfev=300)


        rvec = res.x

    return Rrot.from_rotvec(rvec).as_matrix(), f