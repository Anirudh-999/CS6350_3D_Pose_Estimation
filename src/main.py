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
from epipolar_geometry import estimate_fundamental, estimate_essential_and_pose, draw_epipolar_lines
from triangulation_estimation import triangulate_with_P
from disparity_estimation import compute_disparity, visualize_disparity
from pose_estimation import estimate_pose_from_points

# LEFT_IMG_PATH  = "../images/img2.png"      # path to left image
# RIGHT_IMG_PATH = "../images/img1.png"      # path to right image

# LEFT_IMG_PATH  = "../images/img3.jpeg"      # path to left image
# RIGHT_IMG_PATH = "../images/img4.jpeg"      # path to right image

CALIB_PATH     = "calib.txt"      # optional KITTI-style calib (P0/P1 or K)
YOLO_WEIGHTS   = "yolov8l.pt"     # optional (if ultralytics installed)
OUTPUT_DIR     = "./output"       # where we save visualizations
USE_GPU        = True             # preference flag


os.makedirs(OUTPUT_DIR, exist_ok=True)



# _SPG_AVAILABLE = False
# try:
#     import superpoint 
#     import superglue
#     _SPG_AVAILABLE = True
# except Exception:
#     _SPG_AVAILABLE = False



# print(f"YOLO available: {_YOLO_AVAILABLE}, MiDaS: {_MIDAS_AVAILABLE}, RAFT: {_RRAFT_AVAILABLE}, SPG: {_SPG_AVAILABLE}, Open3D: {_O3D_AVAILABLE}")



def run_pipeline(left_path = None, right_path=None, calib_path=CALIB_PATH, yolo_weights=YOLO_WEIGHTS):
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
        warn("No calibration provided. Will attempt approximations / essential matrix (scale ambiguous).")

    # Preprocess for detection & features
    L_enh = enhance_contrast(L)
    R_enh = enhance_contrast(R)
    show_and_save(L_enh, "Left Enhanced", "03_left_enhanced.png")
    show_and_save(R_enh, "Right Enhanced", "04_right_enhanced.png")

    # Detection
    boxesL, clsL, confL = detect_yolo(L_enh, yolo_weights) if _YOLO_AVAILABLE and yolo_weights and os.path.exists(yolo_weights) else (np.empty((0,4)), np.array([]), np.array([]))
    boxesR, clsR, confR = detect_yolo(R_enh, yolo_weights) if _YOLO_AVAILABLE and yolo_weights and os.path.exists(yolo_weights) else (np.empty((0,4)), np.array([]), np.array([]))

    # Draw detections
    def draw_boxes(img, boxes, color=(0,255,0)):
        out = img.copy()
        for i,b in enumerate(boxes):
            x1,y1,x2,y2 = map(int, b)
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        return out
    show_and_save(draw_boxes(L, boxesL), "Left Detections", "05_left_dets.png")
    show_and_save(draw_boxes(R, boxesR), "Right Detections", "06_right_dets.png")

    # Decide per-object or global
    use_per_object = False
    matches_boxes = []

    if boxesL.shape[0] > 0 and boxesR.shape[0] > 0:
        # match by IoU + class if available
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
        warn("No detections found on one or both images; using whole-image matching.")

    results = []
    # A helper to process a region pair (roiL, roiR) with origins
    def process_region(roiL, roiR, originL=(0,0), originR=(0,0), R_ref=None):
        res = {}
        # features
        kpsL, descL = extract_features_sift(roiL)
        kpsR, descR = extract_features_sift(roiR)
        visualize_keypoints(roiL, kpsL, "07_roiL_kps.png", "ROI Left Keypoints")
        visualize_keypoints(roiR, kpsR, "08_roiR_kps.png", "ROI Right Keypoints")
        # matches
        matches = match_descriptors_flann(descL, descR)
        draw_matches(roiL, kpsL, roiR, kpsR, matches, max_matches=300, fname="09_roi_matches.png", title="ROI Matches")
        if len(matches) < 8:
            warn(f"Only {len(matches)} good matches in ROI; may be insufficient.")
        # build point arrays (global coords)
        ptsL = np.array([kpsL[m.queryIdx].pt for m in matches]) + np.array(originL)
        ptsR = np.array([kpsR[m.trainIdx].pt for m in matches]) + np.array(originR)
        res['ptsL'] = ptsL
        res['ptsR'] = ptsR
        # F matrix + epipolar visualization
        F, maskF = estimate_fundamental(ptsL, ptsR)
        if F is not None:
            mask_in = maskF.ravel().astype(bool)
            draw_epipolar_lines(L, R, ptsL[mask_in][:30], ptsR[mask_in][:30], F, "10_epilines.png")
        # essential & pose
        K = calib.get('K0') if 'K0' in calib else calib.get('K') if 'K' in calib else None
        R_est, t_est, mask_pose = estimate_essential_and_pose(ptsL, ptsR, K)
        res['R'] = R_est
        res['t'] = t_est
        # triangulate if possible
        # P1 = calib.get('P0') if 'P0' in calib else (np.hstack((K, np.zeros((3,1)))) if K is not None else None)
        # P2 = calib.get('P1') if 'P1' in calib else (np.hstack((K@R_est, K@t_est)) if (K is not None and R_est is not None and t_est is not None) else None)
        # if P1 is not None and P2 is not None and ptsL.shape[0] > 0:
        #     try:
        #         pts3d = triangulate_with_P(ptsL[mask_pose] if mask_pose is not None else ptsL, ptsR[mask_pose] if mask_pose is not None else ptsR, P1, P2)
        #         res['pts3d'] = pts3d
        #         info(f"Triangulated {pts3d.shape[0]} points.")
        #         # visualize 3D XY scatter
        #         if pts3d.shape[0] > 0:
        #             fig = plt.figure(figsize=(6,4))
        #             ax = fig.add_subplot(111, projection='3d')
        #             ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2], s=1)
        #             ax.set_title("Triangulated points")
        #             fname = os.path.join(OUTPUT_DIR, "11_tri_scatter.png")
        #             fig.savefig(fname, dpi=150)
        #             plt.close(fig)
        #             info(f"Saved 3D scatter: {fname}")
        #     except Exception as e:
        #         warn(f"Triangulation failed: {e}")
        # else:
        #     warn("P1/P2 not available or insufficient K/R/t to triangulate metric points.")
        #     res['pts3d'] = np.empty((0,3))
        # yaw estimates
        yaw_fromR = yaw_from_R(res['R'], R_ref) if res['R'] is not None else {}
        # yaw_pca = yaw_from_pointcloud_pca(res['pts3d']) if res['pts3d'].shape[0] > 0 else None
        res['yaw_fromR'] = yaw_fromR
        # res['yaw_pca'] = yaw_pca
        return res

    # Extract features from the images
    kpsL, descL = extract_features_sift(L)
    kpsR, descR = extract_features_sift(R)

    # Match descriptors
    matches = match_descriptors_flann(descL, descR)

    # Select the 4 best matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)[:4]

    # Ensure at least 4 matches are found
    if len(matches) < 4:
        err("Insufficient matches found for pose estimation. At least 4 matches are required.")
        return

    # Map the best 4 matches to keys 'A', 'B', 'C', and 'E'
    points_img1 = {key: kpsL[m.queryIdx].pt for key, m in zip(['A', 'B', 'C', 'E'], matches)}
    points_img2 = {key: kpsR[m.trainIdx].pt for key, m in zip(['A', 'B', 'C', 'E'], matches)}

    # Extract corresponding points for pose estimation
    points_img1 = {f"P{i+1}": kpsL[m.queryIdx].pt for i, m in enumerate(matches)}
    points_img2 = {f"P{i+1}": kpsR[m.trainIdx].pt for i, m in enumerate(matches)}

    # Visualize the selected keypoints
    visualize_keypoints(L, [kpsL[m.queryIdx] for m in matches], fname="selected_keypoints_L.png", title="Selected Keypoints Left")
    visualize_keypoints(R, [kpsR[m.trainIdx] for m in matches], fname="selected_keypoints_R.png", title="Selected Keypoints Right")

    # Log the contents of points_img1 and points_img2 for debugging
    info(f"points_img1: {points_img1}")
    info(f"points_img2: {points_img2}")

    # Ensure all required keys are present in points_img1 and points_img2
    required_keys = {'A', 'B', 'C', 'E'}
    if not required_keys.issubset(points_img1.keys()) or not required_keys.issubset(points_img2.keys()):
        err("Missing required keys in points_img1 or points_img2 for pose estimation. Ensure at least 4 matches are found and mapped to 'A', 'B', 'C', 'E'.")
        

    # Map keys 'P1', 'P2', 'P3', 'P4' to 'A', 'B', 'C', 'E'
    points_img1 = {new_key: points_img1[old_key] for new_key, old_key in zip(['A', 'B', 'C', 'E'], points_img1.keys())}
    points_img2 = {new_key: points_img2[old_key] for new_key, old_key in zip(['A', 'B', 'C', 'E'], points_img2.keys())}

    # Estimate pose
    delta_pose, best_params = estimate_pose_from_points(points_img1, points_img2)

    # Print the predicted rotations
    print("Predicted Rotations:")
    print(f"Δα (Tilt): {delta_pose[0]:.2f}°")
    print(f"Δβ (Yaw): {delta_pose[1]:.2f}°")
    print(f"Δγ (Roll): {delta_pose[2]:.2f}°")

    # Process either per-object or whole-image
    if use_per_object:
        for (iL,iR) in matches_boxes:
            bL = boxesL[iL].astype(int); bR = boxesR[iR].astype(int)
            x1,y1,x2,y2 = map(max, (0,0,0,0), bL); X1,Y1,X2,Y2 = map(min, (L.shape[1],L.shape[0],R.shape[1],R.shape[0]), bR)
            roiL = L[y1:y2, x1:x2]
            roiR = R[Y1:Y2, X1:X2]
            info(f"Processing object ROI L#{iL} size {roiL.shape} / R#{iR} size {roiR.shape}")
            res = process_region(roiL, roiR, (x1,y1), (X1,Y1))
            res['pair'] = (iL,iR)
            results.append(res)
    else:
        info("Processing whole-image global pipeline.")
        # whole-image matching
        res = process_region(L, R, (0,0), (0,0))
        res['pair'] = ('global','global')
        results.append(res)

    # # Dense depth/disparity estimation
    # disp = compute_disparity(L, R)
    # visualize_disparity(disp, "13_disparity.png", "Disparity Map")

    # # If no P matrices but we have estimated R/t for one result, build P1/P2 for triangulation & refine
    # if 'P0' not in calib and results and ('K0' in calib or 'K' in calib):
    #     # if any result has R,t we can form P matrices using K and R,t
    #     K = calib.get('K0') if 'K0' in calib else calib.get('K', None)
    #     for r in results:
    #         if r.get('R') is not None and r.get('t') is not None and K is not None:
    #             P1 = np.hstack((K, np.zeros((3,1))))
    #             P2 = np.hstack((K @ r['R'], K @ r['t']))
    #             # attempt triangulation for the matches we used earlier if any
    #             if 'ptsL' in r and 'ptsR' in r and r['ptsL'].shape[0] > 0:
    #                 pts3d = triangulate_with_P(r['ptsL'][r['ptsL'].shape[0]>0], r['ptsR'][r['ptsR'].shape[0]>0], P1, P2)
    #                 r['pts3d_triangulated'] = pts3d
    #                 info(f"Triangulated extra {pts3d.shape[0]} points using constructed P1/P2.")
    # # Final report
    # info("Final results summary:")
    # for i, r in enumerate(results):
    #     info(f"Result #{i}, pair={r.get('pair')}")
    #     pts3d = r.get('pts3d') if r.get('pts3d') is not None else np.empty((0,3))
    #     info(f"  - triangulated pts: {pts3d.shape[0]}")
    #     if r.get('R') is not None:
    #         info(f"  - Rotation matrix (R):\n{r['R']}")
    #         info(f"  - Translation vector (t):\n{r['t'].T}")
    #         info(f"  - yaw candidates: {r.get('yaw_fromR')}")
    #     elif r.get('yaw_pca') is not None:
    #         info(f"  - yaw (PCA): {r.get('yaw_pca'):.2f} deg")
    #     else:
    #         warn("  - No pose/yaw computed for this result.")

    info(f"Total runtime: {time.time()-t0:.2f}s")
    return results

# # ---------- Run ----------
# if __name__ == "__main__":
#     start_time = time.time()
#     try:
#         results = run_pipeline(LEFT_IMG_PATH, RIGHT_IMG_PATH, CALIB_PATH, YOLO_WEIGHTS)
#     except Exception as e:
#         err(f"Unhandled exception: {e}")
#         raise
#     info(f"Done. Script runtime: {time.time()-start_time:.2f}s")
