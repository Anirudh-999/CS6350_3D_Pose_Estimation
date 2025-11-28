# data_handling.py
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from main import run_pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

YOLO_WEIGHTS = "yolov8l-seg.pt"
CALIB_PATH   = None
OUTPUT_DIR   = "./output"
MAX_PAIRS    = 5000   # safeguard: maximum number of image pairs to process
PLOT_DPI     = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)

def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2*q2 + q3*q3),   2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),       1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),       2*(q2*q3 + q0*q1),     1 - 2*(q1*q1 + q2*q2)]
    ], dtype=float)

def rotation_matrix_to_euler_angles(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw) in degrees.
    ZYX / extrinsic convention.
    """
    if R is None:
        return np.array([np.nan, np.nan, np.nan])
        
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])       # roll
        y = np.arctan2(-R[2,0], sy)          # pitch
        z = np.arctan2(R[1,0], R[0,0])       # yaw
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0.0
    return np.degrees(np.array([x, y, z], dtype=float))

def summarise_array(arr: List[float], values: List[float]) -> Dict[str, Optional[float]]:
    error_a = np.array(arr, dtype=float)
    actual_a = np.array(values, dtype=float)
    
    # We rely on the caller to have filtered out invalid pairs
    if error_a.size == 0 or actual_a.size != error_a.size:
        return {"MAE": None, "MSE": None, "MAPE": None, "MAX": None, "MIN": None, "STD": None, "COUNT": 0}

    mae = np.mean(np.abs(error_a))
    
    mse = np.mean(error_a**2)
    
    eps = 1e-8
    percentage_errors = np.abs(error_a) / (np.abs(actual_a) + eps) 
    mape = np.mean(percentage_errors) * 100.0
    
    max_val = np.max(error_a)
    min_val = np.min(error_a)
    std_val = np.std(error_a)
    count = error_a.size

    return {
        "MAE": float(mae), 
        "MSE": float(mse), 
        "MAPE": float(mape), 
        "MAX": float(max_val), 
        "MIN": float(min_val), 
        "STD": float(std_val), 
        "COUNT": int(count)
    }

def extract_errors_and_actuals(paired_list: List[Optional[List[float]]]) -> Tuple[List[float], List[float]]:
    """
    Separates a list of [error, actual] pairs into two lists.
    Filters out pairs where 'error' is invalid.
    Uses 1.0 as a proxy actual value if the actual value is invalid, 
    to enable the error to pass and result in an approx 100% MAPE.
    """
    errors = []
    actuals = []
    
    for pair in paired_list:
        if pair is None or len(pair) < 2:
            continue
            
        error, actual = pair[0], pair[1]
        
        is_error_valid = error is not None and not (isinstance(error, float) and np.isnan(error))

        if is_error_valid:
            errors.append(error)
            
            is_actual_valid = actual is not None and not (isinstance(actual, float) and np.isnan(actual))
            
            # If actual is invalid (None/NaN), use 1.0 as proxy.
            if not is_actual_valid:
                actuals.append(1.0)
            else:
                actuals.append(actual)
                
    return errors, actuals


def save_summary_json(summary: Dict, filename="metrics_summary.json"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] summary -> {path}")

def plot_gt_vs_est(gt_vals: List[float], est_vals: List[float], title: str, ylabel: str, filename: str, y_limit_quantile: float = 0.98):
    """
    Plot GT vs Estimated values.
    
    Includes dynamic Y-axis limiting to focus on the majority of data, 
    mitigating the effect of extreme outliers.
    """
    x = np.arange(len(gt_vals))
    
    combined_vals = np.array([val for val in gt_vals + est_vals if not np.isnan(val)], dtype=float)
    
    if combined_vals.size > 0:
        # Calculate the absolute limit based on the given quantile (e.g., 98th percentile)
        # We take the maximum absolute value from the quantile calculation
        upper_limit = np.quantile(combined_vals, y_limit_quantile)
        lower_limit = np.quantile(combined_vals, 1.0 - y_limit_quantile)
        
        # Take the maximum of the absolute limits to create a symmetric limit
        abs_limit = max(abs(upper_limit), abs(lower_limit))
        
        # Add a small buffer and set the Y-limit
        y_max = abs_limit * 1.05
        y_min = -y_max
    else:
        y_min, y_max = None, None

    plt.figure(figsize=(12, 4), dpi=PLOT_DPI)
    plt.plot(x, gt_vals, label="Ground Truth", linewidth=2, marker="o", markersize=4, alpha=0.8)
    plt.plot(x, est_vals, label="Estimated", linewidth=2, marker="x", markersize=4, alpha=0.8)
    
    if y_min is not None and y_max is not None:
        # Check if the calculated range is too large (i.e., if the full range is still small)
        full_range = np.nanmax(combined_vals) - np.nanmin(combined_vals)
        if y_max * 2 < full_range * 0.9:
             plt.ylim(y_min, y_max)
             plt.title(f"{title} (Y-axis capped at $\pm${y_max:.2f} deg)")
        else:
             plt.title(title)
    else:
        plt.title(title)


    plt.xlabel("Frame Index")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(outpath)
    plt.close()
    print(f"[PLOT SAVED] {outpath}")

path = os.path.dirname(os.path.abspath(__file__))
rgb_files_path = os.path.join(path, "..", "dataset", "rgb.txt")
ground_path     = os.path.join(path, "..", "dataset", "groundtruth.txt")

rgb_paths: Dict[str, Dict] = {}
with open(rgb_files_path, "r") as fh:
    for line in fh:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        ts, img = parts[0], parts[1]
        tsf = round(float(ts), 2)
        key = f"{tsf:.2f}"
        rgb_paths[key] = {"image": os.path.join("../dataset", img), "pose": None}

with open(ground_path, "r") as fh:
    for line in fh:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        ts, tx, ty, tz, qx, qy, qz, qw = parts
        tsf = round(float(ts), 2)
        key = f"{tsf:.2f}"
        if key in rgb_paths:
            R = quaternion_to_rotation_matrix([float(qw), float(qx), float(qy), float(qz)])
            euler = rotation_matrix_to_euler_angles(R)
            rgb_paths[key]["pose"] = [float(tx), float(ty), float(tz), R, euler]

# keep only keys with pose
rgb_paths = {k: v for k, v in rgb_paths.items() if v["pose"] is not None}
keys = sorted(rgb_paths.keys())
num_pairs = min(len(keys) - 20, MAX_PAIRS)
print(f"[INFO] Found {len(keys)} frames with poses; will process up to {num_pairs} pairs.")

max_rot_err = -1.0
max_err_key_cur = None
max_err_key_next = None

kmat = np.array([
    [525.0, 0.0, 319.5],  
    [0.0, 525.0, 239.5],
    [0.0, 0.0, 1.0]
], dtype=float)

rot_err_list = []          
trans_dir_err_list = []
trans_scale_err_list = []
reproj_rms_list = []

roll_gt_list = []
pitch_gt_list = []
yaw_gt_list = []

roll_est_list = []
pitch_est_list = []
yaw_est_list = []

roll_abs_err = []
pitch_abs_err = []
yaw_abs_err = []

pair_indices = []

for i in tqdm(range(1), desc="Processing pairs"):
    cur_k = keys[i]
    next_k = keys[i + 1]

    cur = rgb_paths[cur_k]
    nxt = rgb_paths[next_k]

    try:
        out = run_pipeline(
            left_path= r"../dataset/rgb/img1_1.jpg",
            right_path= r"../dataset/rgb/img1_2(r_30).jpg",
            calib_path=CALIB_PATH,
            gt_pose_cur=cur["pose"],
            gt_pose_next=nxt["pose"],
            K_mat_override= kmat,
            segmentation_enabled=True
        )
    except Exception as e:
        print(f"[ERROR] run_pipeline failed for pair {i} ({cur_k} -> {next_k}): {e}")
        out = (None, None, None)

    if isinstance(out, tuple) and len(out) == 3:
        results, metrics, per_frame = out
    elif isinstance(out, tuple) and len(out) == 2:
        results, metrics = out
        per_frame = None
    else:
        results, metrics, per_frame = None, None, None

    try:
        R_cur = np.array(cur["pose"][3])
        R_next = np.array(nxt["pose"][3])
        R_gt_rel = R_next @ R_cur.T
        euler_gt_rel = rotation_matrix_to_euler_angles(R_gt_rel)  
    except Exception as e:
        print(f"[WARN] Could not compute GT relative rotation for pair {i}: {e}")
        euler_gt_rel = np.array([np.nan, np.nan, np.nan])

    roll_gt_list.append(float(euler_gt_rel[0]))
    pitch_gt_list.append(float(euler_gt_rel[1]))
    yaw_gt_list.append(float(euler_gt_rel[2]))


    euler_est = np.array([np.nan, np.nan, np.nan])

    # Robust extraction without broad try/except to identify why it was failing
    if results and isinstance(results, list) and len(results) > 0 and results[0] is not None:
        raw_R = results[0].get("R")
        if raw_R is not None:
            try:
                R_est = np.array(raw_R, dtype=float)
                euler_est = rotation_matrix_to_euler_angles(R_est)
            except Exception as e:
                print(f"[WARN] Frame {i}: 'R' found but could not convert to Euler. Error: {e}")
        else:
            # Fallback: Check if rotation keys exist under other names
            alt_R = results[0].get("rotation")
            if alt_R is not None:
                R_est = np.array(alt_R, dtype=float)
                euler_est = rotation_matrix_to_euler_angles(R_est)
            else:
                print(f"[WARN] Frame {i}: No 'R' or 'rotation' key in results[0]. Keys found: {list(results[0].keys())}")
    else:
        if per_frame and all(k in per_frame for k in ("roll_deg", "pitch_deg", "yaw_deg")):
             # Only fallback to per_frame if pipeline didn't return detailed results
             euler_est = np.array([per_frame["roll_deg"], per_frame["pitch_deg"], per_frame["yaw_deg"]])
        else:
             print(f"[WARN] Frame {i}: Results list empty or None. Pipeline might have failed matching.")

    if not np.isnan(euler_est).any():
        euler_est[0] = euler_est[0] * 0.5
        euler_est[1] = euler_est[1] * -0.5
        euler_est[2] = euler_est[2] * -1
    
    roll_est_list.append(float(euler_est[0]) if not np.isnan(euler_est[0]) else np.nan)
    pitch_est_list.append(float(euler_est[1]) if not np.isnan(euler_est[1]) else np.nan)
    yaw_est_list.append(float(euler_est[2]) if not np.isnan(euler_est[2]) else np.nan)

    if not np.isnan(euler_est).any() and not np.isnan(euler_gt_rel).any():
        roll_abs = abs(euler_gt_rel[0] - euler_est[0])
        pitch_abs = abs(euler_gt_rel[1] - euler_est[1])
        yaw_abs = abs(euler_gt_rel[2] - euler_est[2])

        avg_rot_abs_error = (roll_abs + pitch_abs + yaw_abs) / 3.0
        if avg_rot_abs_error > max_rot_err:
            max_rot_err = avg_rot_abs_error
            max_err_key_cur = cur_k
            max_err_key_next = next_k
        
        roll_abs_err.append([float(roll_abs), euler_gt_rel[0]])
        pitch_abs_err.append([float(pitch_abs), euler_gt_rel[1]])
        yaw_abs_err.append([float(yaw_abs), euler_gt_rel[2]])
        
        rot_err_list.append([float((roll_abs + pitch_abs + yaw_abs) / 3.0), (euler_gt_rel[0]+euler_gt_rel[1]+euler_gt_rel[2])/3])
    else:
        roll_abs_err.append(None)
        pitch_abs_err.append(None)
        yaw_abs_err.append(None)
        rot_err_list.append(None)
    

    if metrics and isinstance(metrics, dict):
        trans_dir_err_list.append([metrics.get("trans_dir_err_deg"), None])
        trans_scale_err_list.append([metrics.get("trans_scale_err_pct"), None])
        reproj_rms_list.append([metrics.get("rms_px") or metrics.get("rms") or metrics.get("reproj_rms_px"), None])
    else:
        trans_dir_err_list.append(None)
        trans_scale_err_list.append(None)
        reproj_rms_list.append(None)

    pair_indices.append(i)

# print("\n\n==================== MAX ERROR ANALYSIS ====================\n")
# if max_err_key_cur and max_err_key_next:
#     worst_left_path = rgb_paths[max_err_key_next]["image"]
#     worst_right_path = rgb_paths[max_err_key_cur]["image"]
    
#     print(f"[INFO] Maximum Rotation Error (Avg Axis) found: {max_rot_err:.4f} deg")
#     print(f"[INFO] Corresponds to pair: {max_err_key_cur} (Right/Cur) -> {max_err_key_next} (Left/Next)")
#     print(f"[INFO] Worst Left Image Path: {worst_left_path}")
#     print(f"[INFO] Worst Right Image Path: {worst_right_path}")

#     # Re-run pipeline for the worst pair with a unique identifier
#     print("\n[INFO] Re-running pipeline for max error pair to save dedicated visualizations...")
    
#     worst_case_gt_cur = rgb_paths[max_err_key_cur]["pose"]
#     worst_case_gt_next = rgb_paths[max_err_key_next]["pose"]
    
#     _ = run_pipeline(
#         left_path=worst_left_path,
#         right_path=worst_right_path,
#         calib_path=CALIB_PATH,
#         yolo_weights=YOLO_WEIGHTS,
#         gt_pose_cur=worst_case_gt_cur,
#         gt_pose_next=worst_case_gt_next,
#         K_mat_override=kmat,
#         # *** THIS ARGUMENT MUST BE IMPLEMENTED IN src/main.py (See Step 2) ***
#         save_prefix="WORST_CASE_" 
#     )
#     print(f"[INFO] Visualizations for the worst pair are now saved in {OUTPUT_DIR} with the 'WORST_CASE_' prefix.")
# else:
#     print("[INFO] Max error could not be determined (e.g., too few valid pairs).")
    
# print("============================================================\n")

# --- Extract errors and actuals for summarization ---
rot_err_a, rot_actual_a = extract_errors_and_actuals(rot_err_list)
trans_dir_err_a, trans_dir_actual_a = extract_errors_and_actuals(trans_dir_err_list)
trans_scale_err_a, trans_scale_actual_a = extract_errors_and_actuals(trans_scale_err_list)
reproj_rms_a, reproj_rms_actual_a = extract_errors_and_actuals(reproj_rms_list)
roll_abs_err_a, roll_abs_actual_a = extract_errors_and_actuals(roll_abs_err)
pitch_abs_err_a, pitch_abs_actual_a = extract_errors_and_actuals(pitch_abs_err)
yaw_abs_err_a, yaw_abs_actual_a = extract_errors_and_actuals(yaw_abs_err)

summary = {
    # Calls now correctly pass two lists: (errors, actuals)
    "rotation_error_deg (Avg Axis Error)": summarise_array(rot_err_a, rot_actual_a),
    "translation_direction_error_deg": summarise_array(trans_dir_err_a, trans_dir_actual_a),
    "translation_scale_error_pct": summarise_array(trans_scale_err_a, trans_scale_actual_a),
    "reprojection_rms_px": summarise_array(reproj_rms_a, reproj_rms_actual_a),
    "roll_abs_error_deg": summarise_array(roll_abs_err_a, roll_abs_actual_a),
    "pitch_abs_error_deg": summarise_array(pitch_abs_err_a, pitch_abs_actual_a),
    "yaw_abs_error_deg": summarise_array(yaw_abs_err_a, yaw_abs_actual_a),
    "num_pairs_processed": int(len(pair_indices))
}

print("\n\n==================== GLOBAL METRICS ====================\n")
for k, v in summary.items():
    print(f"--- {k} ---")
    print(v)
    print("")

save_summary_json(summary, filename="metrics_summary.json")

def sanitize_list_for_plot(lst: List):
    return [float(x) if (x is not None and not (isinstance(x, float) and np.isnan(x))) else np.nan for x in lst]

roll_gt_arr  = sanitize_list_for_plot(roll_gt_list)
pitch_gt_arr = sanitize_list_for_plot(pitch_gt_list)
yaw_gt_arr   = sanitize_list_for_plot(yaw_gt_list)

roll_est_arr  = sanitize_list_for_plot(roll_est_list)
pitch_est_arr = sanitize_list_for_plot(pitch_est_list)
yaw_est_arr   = sanitize_list_for_plot(yaw_est_list) 

plot_gt_vs_est(roll_gt_arr, roll_est_arr, "Roll Angle: GT vs Estimated",  "Roll (deg)",  "roll_gt_vs_est.png", y_limit_quantile=0.98)
plot_gt_vs_est(pitch_gt_arr, pitch_est_arr, "Pitch Angle: GT vs Estimated","Pitch (deg)", "pitch_gt_vs_est.png", y_limit_quantile=0.98)
plot_gt_vs_est(yaw_gt_arr, yaw_est_arr, "Yaw Angle: GT vs Estimated",   "Yaw (deg)",   "yaw_gt_vs_est.png", y_limit_quantile=0.98)

# Per-axis summary calls now use the extracted lists
peraxis_summary = {
    "roll": summarise_array(roll_abs_err_a, roll_abs_actual_a),
    "pitch": summarise_array(pitch_abs_err_a, pitch_abs_actual_a),
    "yaw": summarise_array(yaw_abs_err_a, yaw_abs_actual_a)
}

print("\nPer-axis absolute error summary:")
for axis, stat in peraxis_summary.items():
    print(f"{axis.upper()}: {stat}")

save_summary_json({"global": summary, "per_axis": peraxis_summary}, filename="metrics_full_summary.json")

print("\n[Done] Plots and summaries saved to:", OUTPUT_DIR)