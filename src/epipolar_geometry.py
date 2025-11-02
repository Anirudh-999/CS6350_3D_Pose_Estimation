import cv2
import numpy as np
from custom_logging import info, warn, show_and_save

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
        # heuristic K (f ~ width)
        w = int(np.max(pts1[:,0]) * 2) if pts1.size else 640
        fx = fy = float(w)
        cx = cy = w/2.0
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        warn("No intrinsics K: approximating; pose will be scale-ambiguous.")
    E, maskE = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    if E is None:
        warn("Essential estimation failed.")
        return None, None, None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    info(f"Recovered pose with {np.sum(mask_pose)} inliers (of {len(pts1)})")
    return R, t, mask_pose
