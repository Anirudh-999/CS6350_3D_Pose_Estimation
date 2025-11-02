import cv2
import numpy as np
from custom_logging import info, warn, show_and_save

_MIDAS_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
    try:
        _MIDAS_AVAILABLE = True
    except Exception:
        _MIDAS_AVAILABLE = False
except Exception:
    torch = None
    _TORCH_AVAILABLE = False
    _MIDAS_AVAILABLE = False

_RRAFT_AVAILABLE = False
try:
    import raft_stereo 
    _RRAFT_AVAILABLE = True
except Exception:
    _RRAFT_AVAILABLE = False


def estimate_depth_midas(img_rgb, device='cuda' if torch and torch.cuda.is_available() else 'cpu'):
    """
    Attempts to use MiDaS via torch.hub. This will download model if not available.
    Returns depth map (H,W) as float32 (higher -> farther or closer depending on model — normalized).
    """
    if not _TORCH_AVAILABLE:
        warn("PyTorch not available; cannot run MiDaS.")
        return None
    try:
        info("Using MiDaS for monocular depth estimation.")
        model_type = "DPT_Large"
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        midas.to(device).eval()
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        img_in = transform(img_rgb).to(device)

        with torch.no_grad():
            prediction = midas(img_in)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth = prediction.cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            info("MiDaS depth estimated successfully.")
            return depth.astype(np.float32)
    except Exception as e:
        warn(f"MiDaS depth estimation failed: {e}")
        return None

def estimate_disparity_sgbm(left_rgb, right_rgb):
    """
    Compute disparity using StereoSGBM with improved parameters and logging.
    """
    try:
        info("Using StereoSGBM for disparity estimation.")
        left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)

        # Improved SGBM parameters
        window_size = 5
        min_disp = 0
        num_disp = 192  # Must be divisible by 16
        matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disp16 = matcher.compute(left_gray, right_gray).astype(np.float32)
        disp = disp16 / 16.0
        disp[disp <= 0] = np.nan  # Mark invalid disparities as NaN
        info("StereoSGBM disparity computed successfully.")
        return disp
    except Exception as e:
        warn(f"StereoSGBM failed: {e}")
        return None


def estimate_disparity_raft(left_rgb, right_rgb):
    """
    Compute disparity using RAFT if available.
    """
    if not _RRAFT_AVAILABLE:
        warn("RAFT not available. Skipping RAFT disparity estimation.")
        return None
    try:
        info("Using RAFT for disparity estimation.")
        disp = raft_stereo.predict_disparity(left_rgb, right_rgb)
        info("RAFT disparity computed successfully.")
        return disp
    except Exception as e:
        warn(f"RAFT disparity estimation failed: {e}")
        return None


def compute_disparity(left_rgb, right_rgb):
    """
    Wrapper to compute disparity using the best available method.
    Logs the chosen method and saves intermediate results.
    """
    disp = None

    # Try RAFT first
    if _RRAFT_AVAILABLE:
        disp = estimate_disparity_raft(left_rgb, right_rgb)
        if disp is not None:
            info("Disparity computed using RAFT.")
            return disp

    # Fallback to SGBM
    disp = estimate_disparity_sgbm(left_rgb, right_rgb)
    if disp is not None:
        info("Disparity computed using StereoSGBM.")
        return disp

    # Fallback to MiDaS (monocular depth)
    if _MIDAS_AVAILABLE:
        depthL = estimate_depth_midas(left_rgb)
        depthR = estimate_depth_midas(right_rgb)
        if depthL is not None and depthR is not None:
            # Resize depthR to match depthL
            if depthL.shape != depthR.shape:
                warn(f"Depth maps have mismatched shapes: {depthL.shape} vs {depthR.shape}. Resizing depthR.")
                depthR = cv2.resize(depthR, (depthL.shape[1], depthL.shape[0]), interpolation=cv2.INTER_LINEAR)
            disp = depthL - depthR  # Approximate disparity from depth difference
            info("Disparity approximated using MiDaS depth maps.")
            return disp

    warn("No valid disparity method succeeded.")
    return None


def visualize_disparity(disp, fname="disparity.png", title="Disparity Map"):
    """
    Visualize and save the disparity map.
    """
    if disp is None:
        warn("No disparity map to visualize.")
        return

    disp_vis = disp.copy()
    disp_vis[np.isnan(disp_vis)] = np.nanmin(disp_vis[np.isfinite(disp_vis)]) if np.any(np.isfinite(disp_vis)) else 0
    disp_vis = disp_vis - np.nanmin(disp_vis)
    if np.nanmax(disp_vis) > 0:
        disp_vis = (disp_vis / np.nanmax(disp_vis) * 255).astype(np.uint8)
    else:
        disp_vis = disp_vis.astype(np.uint8)

    disp_col = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
    show_and_save(disp_col, title, fname)