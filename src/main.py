import os
import sys
import math
import time
import warnings
from typing import Optional, Tuple, Dict, Any, List
from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from custom_logging import info, warn, show_and_save, err

from input_output import load_image_rgb, load_calibration_kitti_like, yaw_from_R, yaw_from_pointcloud_pca, visualize_keypoints
from preprocessing import enhance_contrast, detect_yolo, _YOLO_AVAILABLE
from feature_extraction_maping import extract_features_orb, match_descriptors_flann, draw_matches, extract_features_sift

# --- IMPORT OUR NEW/MOVED FUNCTIONS ---
from epipolar_geometry import estimate_fundamental, self_calibrate_and_find_pose, draw_epipolar_lines, evaluate_pose_accuracy, rotation_matrix_to_euler_angles


# --- (Constants remain the same) ---
CALIB_PATH     = "calib.txt"
YOLO_WEIGHTS   = "yolov8l.pt"
OUTPUT_DIR     = "./output"
USE_GPU        = True

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_pipeline(left_path = None, right_path=None, calib_path=CALIB_PATH, yolo_weights=YOLO_WEIGHTS, gt_pose_cur = None, gt_pose_next = None):
    t0 = time.time()
    
    if os.path.exists(left_path) and os.path.exists(right_path):
        info(f"Running pipeline on images:\n L: {left_path}\n R: {right_path}")
    else:
        err("Left or Right image path does not exist.")
        return
    
    L = load_image_rgb(left_path)
    R = load_image_rgb(right_path)
    info(f"Loaded images: L {L.shape} , R {R.shape}")

    show_and_save(L, "Left Image", "01_left.png", OUTPUT_DIR)
    show_and_save(R, "Right Image", "02_right.png", OUTPUT_DIR)

    calib = {}
    if calib_path and os.path.exists(calib_path):
        calib = load_calibration_kitti_like(calib_path)
    else:
        warn("No calibration provided. Will attempt self-calibration.")

    L_enh = enhance_contrast(L)
    R_enh = enhance_contrast(R)
    
    boxesL, clsL, confL = detect_yolo(L_enh, yolo_weights) if _YOLO_AVAILABLE and yolo_weights and os.path.exists(yolo_weights) else (np.empty((0,4)), np.array([]), np.array([]))
    boxesR, clsR, confR = detect_yolo(R_enh, yolo_weights) if _YOLO_AVAILABLE and yolo_weights and os.path.exists(yolo_weights) else (np.empty((0,4)), np.array([]), np.array([]))

    # --- (Box drawing logic is fine) ---
    def draw_boxes(img, boxes, color=(0,255,0)):
        out = img.copy()
        for i,b in enumerate(boxes):
            x1,y1,x2,y2 = map(int, b)
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        return out
    show_and_save(draw_boxes(L, boxesL), "Left Detections", "05_left_dets.png")
    show_and_save(draw_boxes(R, boxesR), "Right Detections", "06_right_dets.png")


    # --- (Box matching logic is fine) ---
    use_per_object = False
    matches_boxes = []
    if boxesL.shape[0] > 0 and boxesR.shape[0] > 0:
        def iou(a,b):
            ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
            xi1, yi1 = max(ax1,bx1), max(ay1,by1)
            xi2, yi2 = min(ax2,bx2), min(ay2,by2)
            iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
            inter = iw*ih
            aarea = max(0, ax2-ax1)*max(0, ay2-ay1)
            barea = max(0, bx2-bx1)*max(0, by2-by1)
            union = aarea + barea - inter
            return inter/union if union>0 else 0.0
        used = set()
        for i,bl in enumerate(boxesL):
            bestj=-1; bestv=0.0
            for j,br in enumerate(boxesR):
                if j in used: continue
                v = iou(bl,br)
                if v>bestv:
                    bestv=v; bestj=j
            if bestj>=0 and bestv>0.2:
                matches_boxes.append((i,bestj))
                used.add(bestj)
        if matches_boxes:
            use_per_object = True
            info(f"Found {len(matches_boxes)} matched detection pairs.")
        else:
            warn("No matched detection boxes; falling back to global matching.")
    else:
        warn("No detections found; using whole-image matching.")

    results = []
    
    # --- THIS IS THE NEW, CORRECTED process_region FUNCTION ---
    def process_region(roiL, roiR, originL=(0,0), originR=(0,0), R_ref=None, obj_class="unknown"):
        res = {}
        
        # --- 1. Features & Matching ---
        kpsL, descL = extract_features_sift(roiL)
        kpsR, descR = extract_features_sift(roiR)
        visualize_keypoints(roiL, kpsL, f"07_roiL_kps_{obj_class}.png", f"ROI Left Keypoints ({obj_class})")
        visualize_keypoints(roiR, kpsR, f"08_roiR_kps_{obj_class}.png", f"ROI Right Keypoints ({obj_class})")
        
        matches = match_descriptors_flann(descL, descR)
        draw_matches(roiL, kpsL, roiR, kpsR, matches, max_matches=300, fname=f"09_roi_matches_{obj_class}.png", title=f"ROI Matches ({obj_class})")
        
        if len(matches) < 20: # Need more points for H+F
            warn(f"Only {len(matches)} good matches in ROI; may be insufficient.")
            return res
            
        # Build point arrays (global coords)
        ptsL = np.array([kpsL[m.queryIdx].pt for m in matches]) + np.array(originL)
        ptsR = np.array([kpsR[m.trainIdx].pt for m in matches]) + np.array(originR)
        res['ptsL'] = ptsL
        res['ptsR'] = ptsR

        # --- 2. Call our new self-calibration and pose function ---
        K_guess = calib.get('K0') if 'K0' in calib else calib.get('K') # Use calib if present
        
        # Use full image shape L.shape for heuristics inside the function
        R_est, t_est, K_est, non_planar_mask = self_calibrate_and_find_pose(ptsL, ptsR, K_guess, L.shape)

        res['R'] = R_est
        res['t'] = t_est # This 't' is a unit vector
        res['K'] = K_est # This is our (potentially) optimized K
        
        # --- 3. Visualization ---
        if non_planar_mask is not None and len(ptsL[non_planar_mask]) > 8:
            ptsL_nonplanar = ptsL[non_planar_mask]
            ptsR_nonplanar = ptsR[non_planar_mask]
            F_vis, _ = estimate_fundamental(ptsL_nonplanar, ptsR_nonplanar) # Get F for vis
            if F_vis is not None:
                draw_epipolar_lines(L, R, ptsL_nonplanar[:30], ptsR_nonplanar[:30], F_vis, f"10_epilines_nonplanar_{obj_class}.png")

        yaw_fromR = yaw_from_R(res['R'], R_ref) if res['R'] is not None else {}
        res['yaw_fromR'] = yaw_fromR
        return res

    # --- (This main logic is fine as-is) ---
    if use_per_object:
        for (iL,iR) in matches_boxes:
            bL = boxesL[iL].astype(int); bR = boxesR[iR].astype(int)
            x1,y1,x2,y2 = map(max, (0,0,0,0), bL); X1,Y1,X2,Y2 = bR # Simplified
            roiL = L[y1:y2, x1:x2]
            roiR = R[Y1:Y2, X1:X2]
            obj_class = clsL[iL] if iL < len(clsL) else "Unknown"
            info(f"Processing object ROI L#{iL} (class: {obj_class}) size {roiL.shape} / R#{iR} size {roiR.shape}")
            res = process_region(roiL, roiR, (x1,y1), (X1,Y1), obj_class=obj_class)
            res['pair'] = (iL,iR)
            results.append(res)
    else:
        info("Processing whole-image global pipeline.")
        res = process_region(L, R, (0,0), (0,0), obj_class="global")
        res['pair'] = ('global','global')
        results.append(res)
    
    # --- DELETED all the old 4-point logic ---

    # --- NEW, CORRECTED Pose Evaluation Block ---
    try:
        if gt_pose_next is not None and gt_pose_cur is not None and results:
            res = results[0] # Get the first result
            
            # --- 1. Calculate Ground Truth R and t ---
            R_next = np.array(gt_pose_next[3])
            R_cur  = np.array(gt_pose_cur[3])
            t_next_abs = np.array(gt_pose_next[4]) # Absolute position from GT
            t_cur_abs  = np.array(gt_pose_cur[4]) # Absolute position from GT
            
            R_gt_rel = R_next @ R_cur.T
            t_gt_rel = (t_next_abs - R_gt_rel @ t_cur_abs).reshape(3,1)

            if 'R' in res and 't' in res and res['R'] is not None and res['t'] is not None:
                R_est = res['R']
                t_est_unit = res['t'] # This is the unit vector from recoverPose
                
                # --- 2. Get True Scale from GT ---
                true_scale = np.linalg.norm(t_gt_rel)
                
                # --- 3. Apply Scale to our Estimate ---
                # Also, we check the sign. t_est can be +/-.
                # We check which direction is closer to the ground truth.
                t_est_scaled_v1 = t_est_unit * true_scale
                t_est_scaled_v2 = -t_est_unit * true_scale
                
                err_v1 = np.linalg.norm(t_est_scaled_v1 - t_gt_rel)
                err_v2 = np.linalg.norm(t_est_scaled_v2 - t_gt_rel)
                
                t_est_scaled = t_est_scaled_v1 if err_v1 < err_v2 else t_est_scaled_v2
                
                # --- 4. Convert R to Euler Angles ---
                euler_gt = rotation_matrix_to_euler_angles(R_gt_rel)
                euler_est = rotation_matrix_to_euler_angles(R_est)

                print("\n--- Rotation Comparison (Roll, Pitch, Yaw) ---")
                print(f"Ground Truth: {euler_gt[0]:.2f}°, {euler_gt[1]:.2f}°, {euler_gt[2]:.2f}°")
                print(f"Estimated:    {euler_est[0]:.2f}°, {euler_est[1]:.2f}°, {euler_est[2]:.2f}°")
                
                print("\n--- Translation Comparison (Scaled) ---")
                print(f"Ground Truth t: {t_gt_rel.T} (Scale: {true_scale:.4f})")
                print(f"Estimated t:    {t_est_scaled.T}")

                # --- 5. Evaluate using the SCALED t ---
                K_eval = res.get('K') # Use our new estimated K
                if K_eval is None: # Fallback if K wasn't estimated
                   K_eval = calib.get('K0', calib.get('K')) 
                   
                metrics = evaluate_pose_accuracy(R_est, t_est_scaled, R_gt_rel, t_gt_rel, res.get('ptsL'), res.get('ptsR'), K_eval)
                print("Pose Evaluation Metrics:")
                print(metrics)
            else:
                warn("Pose estimation (R or t) was None, cannot evaluate.")
    except Exception as e:
        warn(f"Could not evaluate ground-truth pose: {e}")
        import traceback
        traceback.print_exc()

    info(f"Total runtime: {time.time()-t0:.2f}s")
    return results