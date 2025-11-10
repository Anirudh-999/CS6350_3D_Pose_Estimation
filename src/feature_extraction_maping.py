import cv2
import numpy as np
from custom_logging import info, warn, show_and_save


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