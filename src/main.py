# main.py
import os
import time
import numpy as np
import cv2

from custom_logging import LOGGING_ENABLED, info, warn, err, show_and_save

# Project helpers (you already have these in your repo)
from input_output import load_image_rgb, load_calibration_kitti_like, yaw_from_R, visualize_keypoints
from preprocessing import enhance_contrast, segment_and_get_largest_box
from feature_extraction_maping import (
    match_features_loftr,
    loftr_on_edges,
    extract_features_sift,
    match_descriptors_flann,
    sift_on_edges,
    draw_matches,
)
from epipolar_geometry import (
    evaluate_pose_accuracy,
    rotation_matrix_to_euler_angles,
)

OUTPUT_DIR = "./output"
YOLO_WEIGHTS = "yolov8l-seg.pt"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def centroid_from_mask(mask_img):
    """Compute centroid of non-zero pixels in mask (RGB or single channel)."""
    if mask_img is None:
        return None
    gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY) if mask_img.ndim == 3 else mask_img
    ys, xs = np.nonzero(gray)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def pick_best_homography_candidate(Rs, Ts, normals, ptsL, ptsR, K):
    """
    Heuristic to pick the best homography decomposition candidate.
    Prefers rotations with det ~ +1 and normals with larger absolute z,
    then slightly prefers larger translation magnitude.
    """
    best_idx = None
    best_score = 1e9
    for i, R in enumerate(Rs):
        try:
            detR = float(np.linalg.det(R))
        except Exception:
            detR = -1.0
        if detR < 0.0:
            # discard flipped parity where possible
            continue
        n = normals[i] if i < len(normals) else np.array([0.0, 0.0, 0.0])
        nz = float(n[2]) if n is not None else 0.0
        t = Ts[i] if i < len(Ts) else np.zeros(3)
        tnorm = float(np.linalg.norm(t))
        # score: prefer large |nz|, prefer larger tnorm modestly
        score = -abs(nz) - 0.01 * tnorm
        if score < best_score:
            best_score = score
            best_idx = i
    if best_idx is None and len(Rs) > 0:
        best_idx = 0
    return best_idx


def safe_show_and_save(img_rgb, title, fname):
    """Wrapper: only call show_and_save if provided and fname is str."""
    try:
        if fname is None:
            return
        show_and_save(img_rgb, title, fname)
    except Exception:
        # Don't crash the pipeline on visualization failures
        if LOGGING_ENABLED:
            warn(f"Could not save {fname}")


def run_pipeline(left_path, right_path, calib_path=None, yolo_weights=YOLO_WEIGHTS,
                 gt_pose_cur=None, gt_pose_next=None, save_vis=True):
    """
    Main pipeline:
      - load images
      - segmentation -> ROI
      - matching chain (LoFTR-on-edges -> LoFTR -> SIFT-on-edges -> SIFT)
      - homography (RANSAC) + decompose
      - choose best R,t
      - evaluate vs GT if provided

    Returns:
      results_list, metrics_dict, per_frame_metrics_dict

    per_frame_metrics_dict contains:
      { 'roll_err', 'pitch_err', 'yaw_err',
        'rot_err_deg', 'trans_dir_err_deg', 'trans_scale_err_pct', 'rms_px' }
    """
    t0 = time.time()
    if not (os.path.exists(left_path) and os.path.exists(right_path)):
        if LOGGING_ENABLED:
            err("Left or Right image path does not exist.")
        return None, None, None

    L = load_image_rgb(left_path)
    R = load_image_rgb(right_path)
    if LOGGING_ENABLED:
        info(f"Loaded images: L {L.shape}, R {R.shape}")

    # quick saves
    if save_vis:
        safe_show_and_save(L, "Left Image", os.path.join(OUTPUT_DIR, "01_left.png"))
        safe_show_and_save(R, "Right Image", os.path.join(OUTPUT_DIR, "02_right.png"))

    # calibration guess
    Kmat = None
    if calib_path and os.path.exists(calib_path):
        try:
            calib = load_calibration_kitti_like(calib_path)
            Kmat = calib.get("K0") or calib.get("K")
            if LOGGING_ENABLED:
                info("Using provided calibration.")
        except Exception:
            Kmat = None

    if Kmat is None:
        h, w = L.shape[:2]
        f_guess = float(max(w, h))
        cx, cy = w / 2.0, h / 2.0
        Kmat = np.array([[f_guess, 0, cx], [0, f_guess, cy], [0, 0, 1]], dtype=float)
        if LOGGING_ENABLED:
            warn("No calibration provided. Will use heuristic focal ~ width.")

    # preprocessing
    L_enh = enhance_contrast(L)
    R_enh = enhance_contrast(R)

    # segmentation (primary)
    try:
        boxesL, clsL, confL, masked_L = segment_and_get_largest_box(L_enh, yolo_weights,
                                                                    save_mask_path=os.path.join(OUTPUT_DIR, "05_left_seg_mask.png"))
        boxesR, clsR, confR, masked_R = segment_and_get_largest_box(R_enh, yolo_weights,
                                                                    save_mask_path=os.path.join(OUTPUT_DIR, "06_right_seg_mask.png"))
    except Exception as e:
        if LOGGING_ENABLED:
            warn(f"Segmentation failed: {e}. Falling back to whole-image.")
        masked_L, masked_R = L_enh, R_enh
        boxesL = np.zeros((0, 4)); boxesR = np.zeros((0, 4))

    # ROI extraction
    def crop_from_box(img, box):
        if getattr(box, "size", 0) == 0 or box is None:
            return img, (0, 0)
        b = [int(v) for v in box]
        x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), min(img.shape[1], b[2]), min(img.shape[0], b[3])
        return img[y1:y2, x1:x2], (x1, y1)

    if len(boxesL) > 0 and len(boxesR) > 0:
        roiL, originL = crop_from_box(masked_L, boxesL[0])
        roiR, originR = crop_from_box(masked_R, boxesR[0])
    else:
        roiL, originL = masked_L, (0, 0)
        roiR, originR = masked_R, (0, 0)

    # centroid shift (image space) - useful as a sanity check translation
    centroid_L = centroid_from_mask(masked_L)
    centroid_R = centroid_from_mask(masked_R)
    centroid_shift_px = None
    if centroid_L is not None and centroid_R is not None:
        centroid_shift_px = centroid_R - centroid_L
        if LOGGING_ENABLED:
            info(f"centroid pixel shift (R - L): {centroid_shift_px}")

    # ---------- Matching chain ----------
    def run_matching(roiL_img, roiR_img):
        # 1) LoFTR on edges
        if LOGGING_ENABLED:
            info("Trying LoFTR on edges...")
        try:
            k1_e, k2_e, m_e = loftr_on_edges(roiL_img, roiR_img, obj_class="roi")
        except Exception as e:
            k1_e, k2_e, m_e = [], [], []
            if LOGGING_ENABLED:
                warn(f"loftr_on_edges failed: {e}")

        if len(m_e) >= 30:
            return k1_e, k2_e, m_e

        # 2) raw LoFTR
        if LOGGING_ENABLED:
            info("Trying raw LoFTR...")
        try:
            k1_r, k2_r, m_r = match_features_loftr(roiL_img, roiR_img)
        except Exception as e:
            k1_r, k2_r, m_r = [], [], []
            if LOGGING_ENABLED:
                warn(f"raw LoFTR failed: {e}")

        if len(m_r) >= 20:
            return k1_r, k2_r, m_r

        # 3) SIFT-on-edges
        if LOGGING_ENABLED:
            info("Trying SIFT-on-edges...")
        try:
            kL_e, dL_e, _ = sift_on_edges(roiL_img)
            kR_e, dR_e, _ = sift_on_edges(roiR_img)
            m_sf = match_descriptors_flann(dL_e, dR_e)
        except Exception as e:
            kL_e, kR_e, m_sf = [], [], []
            if LOGGING_ENABLED:
                warn(f"sift_on_edges failed: {e}")

        if len(m_sf) >= 12:
            return kL_e, kR_e, m_sf

        # 4) plain SIFT+FLANN
        if LOGGING_ENABLED:
            info("Trying plain SIFT+FLANN...")
        try:
            kL_s, dL_s = extract_features_sift(roiL_img)
            kR_s, dR_s = extract_features_sift(roiR_img)
            m_s = match_descriptors_flann(dL_s, dR_s)
        except Exception as e:
            kL_s, kR_s, m_s = [], [], []
            if LOGGING_ENABLED:
                warn(f"plain SIFT fallback failed: {e}")

        if len(m_s) >= 8:
            return kL_s, kR_s, m_s

        return [], [], []

    kps1, kps2, matches = run_matching(roiL, roiR)

    if len(matches) < 8:
        if LOGGING_ENABLED:
            warn(f"Insufficient matches ({len(matches)}). Aborting.")
        return None, None, None

    # Build global image coords arrays
    try:
        ptsL = np.array([kps1[m.queryIdx].pt for m in matches]) + np.array(originL)
        ptsR = np.array([kps2[m.trainIdx].pt for m in matches]) + np.array(originR)
    except Exception as e:
        if LOGGING_ENABLED:
            warn(f"Failed building pts arrays from matches: {e}")
        return None, None, None

    # optional draw matches
    try:
        draw_matches(roiL, kps1, roiR, kps2, matches,
                     fname=os.path.join(OUTPUT_DIR, "roi_matches.png"),
                     title="ROI Matches")
    except Exception:
        pass

    # ---------- Homography ----------
    H, maskH = cv2.findHomography(ptsL, ptsR, cv2.RANSAC, 3.0)
    if H is None:
        if LOGGING_ENABLED:
            warn("findHomography returned None.")
        return None, None, None
    maskH = maskH.ravel().astype(bool)
    inlier_count = int(np.sum(maskH))
    if LOGGING_ENABLED:
        info(f"Homography inliers: {inlier_count}/{len(matches)}")

    # decompose homography (handle opencv variations)
    try:
        decomp = cv2.decomposeHomographyMat(H, Kmat)
        # decomp may contain retval on some versions
        if len(decomp) == 4:
            _, Rs, Ts, normals = decomp
        elif len(decomp) == 3:
            Rs, Ts, normals = decomp
        else:
            if LOGGING_ENABLED:
                warn(f"Unexpected decomposeHomographyMat result len={len(decomp)}")
            return None, None, None
        Rs = list(Rs)
        Ts = list(Ts)
        normals = list(normals)
        if len(Rs) == 0:
            if LOGGING_ENABLED:
                warn("decomposeHomographyMat returned zero candidates.")
            return None, None, None
    except Exception as e:
        if LOGGING_ENABLED:
            warn(f"decomposeHomographyMat failed: {e}")
        return None, None, None

    best_idx = pick_best_homography_candidate(Rs, Ts, normals, ptsL[maskH], ptsR[maskH], Kmat)
    if best_idx is None:
        best_idx = 0
    R_est = Rs[best_idx].astype(float)
    t_est = Ts[best_idx].reshape(3, 1).astype(float)
    normal_est = normals[best_idx].reshape(3, 1).astype(float)

    euler_est = rotation_matrix_to_euler_angles(R_est)

    # compute reprojection RMS (homography projection)
    ptsL_h = np.hstack([ptsL, np.ones((ptsL.shape[0], 1))])
    proj = (H @ ptsL_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    reproj_errs = np.linalg.norm(proj - ptsR, axis=1)
    reproj_rms = float(np.sqrt(np.mean(reproj_errs[maskH]**2))) if np.sum(maskH) > 0 else float(np.sqrt(np.mean(reproj_errs**2)))

    # Prepare results
    res = {
        "R": R_est,
        "t": t_est,
        "normal": normal_est,
        "H": H,
        "K": Kmat,
        "ptsL": ptsL,
        "ptsR": ptsR,
        "inliers_mask": maskH,
        "reproj_rms_px": reproj_rms,
        "centroid_shift_px": centroid_shift_px,
        "inlier_count": inlier_count
    }

    # per-frame metrics placeholder
    per_frame_metrics = {
        "roll_err": None,
        "pitch_err": None,
        "yaw_err": None,
        "rot_err_deg": None,
        "trans_dir_err_deg": None,
        "trans_scale_err_pct": None,
        "rms_px": reproj_rms
    }

    metrics = None
    # Evaluate vs GT if provided
    if gt_pose_cur is not None and gt_pose_next is not None:
        try:
            R_next = np.array(gt_pose_next[3])
            R_cur = np.array(gt_pose_cur[3])
            t_next = np.array([float(x) for x in gt_pose_next[0:3]]).reshape(3, 1)
            t_cur = np.array([float(x) for x in gt_pose_cur[0:3]]).reshape(3, 1)

            R_gt_rel = R_next @ R_cur.T
            t_gt_rel = (t_next - R_gt_rel @ t_cur).reshape(3, 1)

            # scale t_est to gt magnitude (for comparison only)
            scale = float(np.linalg.norm(t_gt_rel) + 1e-12)
            t1 = t_est * scale
            t2 = -t_est * scale
            t_est_scaled = t1 if np.linalg.norm(t1 - t_gt_rel) < np.linalg.norm(t2 - t_gt_rel) else t2

            euler_gt = rotation_matrix_to_euler_angles(R_gt_rel)

            # axis errors
            roll_err = float(abs(euler_gt[0] - euler_est[0]))
            pitch_err = float(abs(euler_gt[1] - euler_est[1]))
            yaw_err = float(abs(euler_gt[2] - euler_est[2]))

            per_frame_metrics.update({
                "roll_err": roll_err,
                "pitch_err": pitch_err,
                "yaw_err": yaw_err,
            })

            # compute full metrics via evaluate_pose_accuracy if available
            try:
                metrics = evaluate_pose_accuracy(R_est, t_est_scaled, R_gt_rel, t_gt_rel, ptsL[maskH], ptsR[maskH], Kmat)
                # ensure numeric types
                for k, v in metrics.items():
                    try:
                        metrics[k] = float(v)
                    except Exception:
                        pass
                # fill per_frame rotational + others if evaluate returns them
                if "rot_err_deg" in metrics:
                    per_frame_metrics["rot_err_deg"] = metrics["rot_err_deg"]
                if "trans_dir_err_deg" in metrics:
                    per_frame_metrics["trans_dir_err_deg"] = metrics["trans_dir_err_deg"]
                if "trans_scale_err_pct" in metrics:
                    per_frame_metrics["trans_scale_err_pct"] = metrics["trans_scale_err_pct"]
                if "rms_px" in metrics:
                    per_frame_metrics["rms_px"] = metrics["rms_px"]
            except Exception:
                # fallback manual simple metrics
                Rdiff = R_est @ R_gt_rel.T
                angle = np.arccos(np.clip((np.trace(Rdiff) - 1) / 2, -1, 1))
                rot_err_deg = float(np.degrees(abs(angle)))
                td1 = (t_est_scaled.ravel() / (np.linalg.norm(t_est_scaled) + 1e-12))
                td2 = (t_gt_rel.ravel() / (np.linalg.norm(t_gt_rel) + 1e-12))
                trans_dir_err_deg = float(np.degrees(np.arccos(np.clip(np.dot(td1, td2), -1, 1))))
                trans_scale_err_pct = float(abs(np.linalg.norm(t_est_scaled) - np.linalg.norm(t_gt_rel)) / (np.linalg.norm(t_gt_rel) + 1e-12) * 100.0)
                metrics = {
                    "rot_err_deg": rot_err_deg,
                    "trans_dir_err_deg": trans_dir_err_deg,
                    "trans_scale_err_pct": trans_scale_err_pct,
                    "rms_px": float(reproj_rms)
                }
                per_frame_metrics.update({
                    "rot_err_deg": rot_err_deg,
                    "trans_dir_err_deg": trans_dir_err_deg,
                    "trans_scale_err_pct": trans_scale_err_pct,
                })

            # Logging
            if LOGGING_ENABLED:
                info("\n=== POSE COMPARISON ===")
                info(f"Ground truth (Euler deg): {np.round(euler_gt, 3)}")
                info(f"Estimated    (Euler deg): {np.round(euler_est, 3)}")
                info(f"GT translation: {t_gt_rel.ravel()}")
                info(f"Estimated translation (scaled): {t_est_scaled.ravel()}")
                info(f"Reproj RMS (px): {reproj_rms}")
                info("========================\n")

        except Exception as e:
            if LOGGING_ENABLED:
                warn(f"Pose evaluation failed: {e}")
            metrics = None

    results = [res]
    if LOGGING_ENABLED:
        info(f"Total runtime: {time.time() - t0:.2f}s")

    return results, metrics, per_frame_metrics
