import cv2
import numpy as np
from custom_logging import info, warn, show_and_save, err

# --- New Imports for LoFTR ---
_LOFTR_AVAILABLE = False
try:
    import torch
    import kornia as K
    import kornia.feature as KF
    _LOFTR_AVAILABLE = True
except Exception as e:
    warn("kornia/torch not installed or failed to import. LoFTR will not be available.")
    _LOFTR_AVAILABLE = False


def extract_features_sift(img_rgb: np.ndarray):
    """
    Extract SIFT features from an RGB image with subpixel refinement.
    returns keypoints and descriptors.

    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
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
    """
    Extract ORB features from an RGB image.
    returns keypoints and descriptors.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(5000)
    kps, desc = orb.detectAndCompute(gray, None)
    return kps, desc


def match_descriptors_flann(desc1, desc2, kps1=None, kps2=None,
                            img1=None, img2=None,
                            ratio=0.75,
                            fun = None):
    """
    Match descriptors using FLANN-based matcher with Lowe's ratio test.
    If keypoints + images are provided, draw the matches inside this function.
    """
    if desc1 is None or desc2 is None:
        return []

    # float descriptors (SIFT)
    if desc1.dtype != np.uint8:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        try:
            matches_knn = matcher.knnMatch(desc1, desc2, k=2)
        except Exception:
            # fallback
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches_knn = bf.knnMatch(desc1, desc2, k=2)
    else:
        # ORB version
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches_knn = bf.knnMatch(desc1, desc2, k=2)

    # ratio test
    good = []
    for m in matches_knn:
        if len(m) >= 2:
            a, b = m[0], m[1]
            if a.distance < ratio * b.distance:
                good.append(a)

    if kps1 is not None and kps2 is not None and img1 is not None and img2 is not None:
        if len(good) > 0:
            img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

            vis = cv2.drawMatches(
                img1_bgr, kps1,
                img2_bgr, kps2,
                good[:200],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            if fun == "canny":
                show_and_save(vis_rgb, "SIFT Matches", "sift_edge_matches.png")
            elif fun == "raw":
                show_and_save(vis_rgb, "SIFT Matches", "sift_raw_matches.png")

    return good


def draw_matches(img1, kps1, img2, kps2, matches, max_matches=200, fname=None, title="Matches"):
    """
    Draw matches between two images and save to disk.
    """
    if len(matches) == 0:
        warn("No matches to draw.")
        return

    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    out = cv2.drawMatches(img1_bgr, kps1, img2_bgr, kps2, matches[:max_matches], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    show_and_save(out_rgb, title, fname)


def _load_torch_image_gray(img_rgb: np.ndarray):
    """Converts a CV2 RGB image to a grayscale torch tensor."""
    import kornia as K  # local import to be safe
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_tensor = K.image_to_tensor(gray, False).float() / 255.0  # (1, 1, H, W)
    return img_tensor


def match_features_loftr(img1_rgb: np.ndarray, img2_rgb: np.ndarray, fun):
    """
    Matches features using LoFTR (kornia). Returns (kps1, kps2, matches)
    where kps1 and kps2 are lists of cv2.KeyPoint and matches is a list
    of cv2.DMatch with corresponding indices.
    """
    if not _LOFTR_AVAILABLE:
        warn("LoFTR not available - returning empty matches.")
        return [], [], []

    try:
        import torch
        import kornia.feature as KF
    except Exception as e:
        warn(f"LoFTR import error: {e}")
        return [], [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model (robustly)
    try:
        matcher = KF.LoFTR(pretrained='indoor').to(device).eval()
    except Exception as e:
        warn(f"Could not instantiate LoFTR: {e}")
        return [], [], []

    # Prepare images (grayscale tensors)
    try:
        img1_t = _load_torch_image_gray(img1_rgb).to(device)
        img2_t = _load_torch_image_gray(img2_rgb).to(device)
        input_dict = {"image0": img1_t, "image1": img2_t}
    except Exception as e:
        warn(f"Failed to prepare images for LoFTR: {e}")
        return [], [], []

    info("Running LoFTR inference...")
    try:
        with torch.no_grad():
            correspondences = matcher(input_dict)
    except Exception as e:
        err(f"LoFTR inference failed: {e}")
        return [], [], []

    # Extract keypoints
    mkpts0 = correspondences.get('keypoints0', None)
    mkpts1 = correspondences.get('keypoints1', None)
    if mkpts0 is None or mkpts1 is None:
        warn("LoFTR returned no keypoints.")
        return [], [], []

    mkpts0 = mkpts0.cpu().numpy()
    mkpts1 = mkpts1.cpu().numpy()

    kps1 = []
    kps2 = []
    matches = []

    # build keypoints and matches with consistent indexing
    for i in range(len(mkpts0)):
        x0, y0 = float(mkpts0[i, 0]), float(mkpts0[i, 1])
        x1, y1 = float(mkpts1[i, 0]), float(mkpts1[i, 1])

        # Correct use: x, y, size
        kp1 = cv2.KeyPoint(x0, y0, 1)
        kp2 = cv2.KeyPoint(x1, y1, 1)

        kps1.append(kp1)
        kps2.append(kp2)

        dm = cv2.DMatch(i, i, 0.0)
        matches.append(dm)

    info(f"LoFTR found {len(matches)} matches.")

    vis_img = None
    if fun == "canny":
        vis_img = draw_matches(
            img1_rgb, kps1,
            img2_rgb, kps2,
            matches,
            max_matches=200,
            fname="loftr_edge_matches.png",
            title="LoFTR Edge Matches"
        )
    elif fun == "raw":
        vis_img = draw_matches(
            img1_rgb, kps1,
            img2_rgb, kps2,
            matches,
            max_matches=200,
            fname="loftr_raw_matches.png",
            title="LoFTR Raw Matches"
        )
    return kps1, kps2, matches

def detect_edges(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Denoise
    blur = cv2.bilateralFilter(gray, 15, 75, 75)

    # 2. Canny
    edges = cv2.Canny(blur, 100, 200)  # adjust as needed

    # 3. Remove small noise
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges


def sift_on_edges(img_rgb):
    """
    Extract SIFT features only on edge pixels detected by Canny.
    
    returns keypoints, descriptors, edges"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(gray, None)

    if desc is None:
        return [], None, edges

    filtered_kps = []
    filtered_desc = []

    for kp, d in zip(kps, desc):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if edges[y, x] > 0:
            filtered_kps.append(kp)
            filtered_desc.append(d)

    if len(filtered_desc) == 0:
        return [], None, edges

    filtered_desc = np.array(filtered_desc)
    return filtered_kps, filtered_desc, edges


def loftr_on_edges(img1_rgb, img2_rgb, obj_class="unknown", output_dir="./output"):
    """
    Run LoFTR on ROI but only keep matches where both points lie on edges.
    Saves:
      - Canny edges
      - keypoints on edges
      - matches on edges
    """

    # 1. Compute Canny edges
    gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

    edges1 = cv2.Canny(gray1, 50, 150)
    edges2 = cv2.Canny(gray2, 50, 150)

    # Save edges
    show_and_save(cv2.cvtColor(edges1, cv2.COLOR_GRAY2RGB),
                  f"Edges L ({obj_class})", 
                  f"edges_left_{obj_class}.png", output_dir)
    show_and_save(cv2.cvtColor(edges2, cv2.COLOR_GRAY2RGB),
                  f"Edges R ({obj_class})", 
                  f"edges_right_{obj_class}.png", output_dir)

    # 2. Run raw LoFTR on the ROI
    kps1_raw, kps2_raw, matches_raw = match_features_loftr(img1_rgb, img2_rgb)
    if len(matches_raw) == 0:
        return [], [], []

    # 3. Apply edge constraint filter
    filtered_kps1 = []
    filtered_kps2 = []
    filtered_matches = []

    for idx, m in enumerate(matches_raw):
        x1, y1 = map(int, kps1_raw[m.queryIdx].pt)
        x2, y2 = map(int, kps2_raw[m.trainIdx].pt)

        if 0 <= y1 < edges1.shape[0] and 0 <= x1 < edges1.shape[1] \
           and 0 <= y2 < edges2.shape[0] and 0 <= x2 < edges2.shape[1]:

            if edges1[y1, x1] > 0 and edges2[y2, x2] > 0:
                filtered_kps1.append(kps1_raw[m.queryIdx])
                filtered_kps2.append(kps2_raw[m.trainIdx])
                filtered_matches.append(
                    cv2.DMatch(len(filtered_kps1)-1, len(filtered_kps2)-1, 0)
                )

    # 4. Save edge-keypoints visualization
    if len(filtered_kps1) > 0:
        draw_matches(img1_rgb, filtered_kps1,
                     img2_rgb, filtered_kps2,
                     filtered_matches,
                     max_matches=300,
                     fname=f"roi_edge_matches_{obj_class}.png",
                     title=f"LoFTR Edge Matches ({obj_class})")

    return filtered_kps1, filtered_kps2, filtered_matches


