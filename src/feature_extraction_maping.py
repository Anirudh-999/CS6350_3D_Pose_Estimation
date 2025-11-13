import cv2
import numpy as np
from custom_logging import info, warn, show_and_save,err

# --- New Imports for LoFTR ---
try:
    import torch
    import kornia as K
    import kornia.feature as KF
    _LOFTR_AVAILABLE = True
except ImportError:
    warn("kornia or torch not installed. LoFTR will not be available.")
    _LOFTR_AVAILABLE = False
# --- End New Imports ---


def extract_features_sift(img_rgb: np.ndarray):

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    try:
        sift = cv2.SIFT_create()
    except Exception:
        sift = cv2.SIFT_create()  

    kps, desc = sift.detectAndCompute(gray, None)

    # Subpixel refinement
    if len(kps) > 0:
        corners = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)
        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
        )
        for i, kp in enumerate(kps):
            kp.pt = tuple(refined_corners[i, 0])

    return kps, desc

def extract_features_orb(img_rgb: np.ndarray):

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(5000)
    kps, desc = orb.detectAndCompute(gray, None)

    return kps, desc

def match_descriptors_flann(desc1, desc2, ratio=0.75):

    if desc1 is None or desc2 is None:
        return []
    
    # FLANN for SIFT (float descriptors)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(desc1, desc2, k=2)

    except Exception:

    # If desc are uint8 (ORB), use BF matcher instead

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(desc1, desc2, k=2)
        matches = knn

    good = []

    for m in matches:
        if len(m) >= 2:
            a, b = m[0], m[1]
            if a.distance < ratio * b.distance:
                good.append(a)
    return good

def draw_matches(img1, kps1, img2, kps2, matches, max_matches=200, fname=None, title="Matches"):

    if len(matches) == 0:
        warn("No matches to draw.")
        return
    
    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    out = cv2.drawMatches(img1_bgr, kps1, img2_bgr, kps2, matches[:max_matches], None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    show_and_save(out_rgb, title, fname)

# --- START: New LoFTR Function ---

def _load_torch_image_gray(img_rgb: np.ndarray):
    """Converts a CV2 RGB image to a grayscale torch tensor."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_tensor = K.image_to_tensor(gray, False).float() / 255.0  # (1, 1, H, W)
    return img_tensor

def match_features_loftr(img1_rgb: np.ndarray, img2_rgb: np.ndarray):
    """
    Matches features using LoFTR.
    Returns (kps1, kps2, matches) to be compatible with draw_matches.
    
    Note: You may want to apply segmentation masks to img1_rgb and img2_rgb
    *before* passing them to this function for best results on textureless objects.
    e.g., img1_masked = cv2.bitwise_and(img1_rgb, img1_rgb, mask=mask1)
    """
    if not _LOFTR_AVAILABLE:
        warn("LoFTR is not available. Skipping match_features_loftr.")
        return [], [], []

    # 1. Setup device and model
    device = K.utils.get_cuda_device_if_available()
    try:
        matcher = KF.LoFTR(pretrained='indoor').to(device).eval()
    except Exception as e:
        warn(f"Could not load LoFTR model. Is kornia installed? Error: {e}")
        return [], [], []

    # 2. Prepare images
    # We convert to grayscale tensor, as LoFTR works on grayscale
    img1_t = _load_torch_image_gray(img1_rgb).to(device)
    img2_t = _load_torch_image_gray(img2_rgb).to(device)
    
    input_dict = {"image0": img1_t, "image1": img2_t}
    info("Running LoFTR inference...")

    # 3. Run inference
    try:
        with torch.no_grad():
            correspondences = matcher(input_dict)
    except Exception as e:
        err(f"LoFTR inference failed. Error: {e}")
        return [], [], []

    # 4. Convert LoFTR output to the (kps1, kps2, matches)
    # format required by your draw_matches function.
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    
    kps1 = []
    kps2 = []
    matches = []
    
    for i in range(len(mkpts0)):
        # Get coords
        x0, y0 = mkpts0[i]
        x1, y1 = mkpts1[i]
        
        # Create KeyPoint objects
        # size=1 is just a placeholder, LoFTR doesn't provide scale
        kp1 = cv2.KeyPoint(x=float(x0), y=float(y0), size=1)
        kp2 = cv2.KeyPoint(x=float(x1), y=float(y1), size=1)
        
        # Create DMatch object
        # queryIdx = index in kps1, trainIdx = index in kps2
        # LoFTR provides direct correspondences, so distance is 0
        # NEW
        match = cv2.DMatch(i, i, 0.0) # Or just (i, i, 0)
        
        kps1.append(kp1)
        kps2.append(kp2)
        matches.append(match)
        
    info(f"LoFTR found {len(matches)} matches.")
    
    return kps1, kps2, matches

# --- END: New LoFTR Function ---