# 3D Pose Change Estimation (Uncalibrated Two-View)

This repository implements a pipeline to estimate **6-DoF relative pose
changes** of a *near‑planar object* using only **two uncalibrated RGB
images**.\
The system is built as part of **CS6350 -- Computer Vision** coursework
at IIT Madras.

------------------------------------------------------------------------

## 🚀 Key Idea

Recovering 3D pose change from two uncalibrated images is difficult
because: - No camera intrinsics (focal length, principal point). -
Translation is only recoverable *up to scale*. - Near-planar,
low‑texture objects break standard feature matchers.

To overcome this, the project uses **segmentation + edge‑aware feature
matching + homography decomposition**.

------------------------------------------------------------------------

## 📦 Pipeline Overview

1.  **Segmentation (FastSAM)**
    -   Extract object mask\
    -   Crop tight ROI\
    -   Remove background clutter
2.  **Edge-Aware Matching**
    -   Compute edges using Canny\
    -   LoFTR matches restricted to edge pixels\
    -   SIFT matches filtered by edge map\
    -   Fallback to raw LoFTR/SIFT if needed
3.  **Homography Estimation**
    -   RANSAC-based H estimation\

    -   Intrinsics approximated as:

            fx = fy = max(W, H)
            cx = W/2, cy = H/2
4.  **Homography Decomposition**
    -   Extract candidate (R, t, n)\
    -   Filter invalid ones\
    -   Select best using heuristics
5.  **Evaluation (Using TUM RGB-D Ground Truth)**
    -   Rotation error\
    -   Translation direction error\
    -   Reprojection RMS

------------------------------------------------------------------------

## 📊 Results Summary

### ✔ Small-frame motion (≈1100 pairs)

-   **Rotation error:** \~0.6° MAE\
-   **Roll & Pitch:** very accurate\
-   **Yaw:** unstable for planar geometry\
-   **Translation direction error:** high (expected for homography)\
-   **Scale:** fundamentally unrecoverable

### ✔ Large-frame motion (≈1000 pairs)

-   **Rotation error:** rises to \~9° MAE\
-   **Failures** occur when homography decomposition selects a wrong
    branch\
-   **Reprojection RMS:** stays low → matching is correct; decomposition
    is ambiguous

------------------------------------------------------------------------

## 🧠 Insights & Limitations

### Strengths

-   Robust edge-aware matching\
-   Accurate roll/pitch rotation for near-planar objects\
-   Stable homographies due to careful matching

### Limitations

-   Translation only up to scale\
-   Yaw poorly observed\
-   Large viewpoint changes cause decomposition ambiguity

------------------------------------------------------------------------

## 🛠 Requirements

-   Python 3.8+
-   OpenCV
-   PyTorch
-   LoFTR model files
-   FastSAM
-   NumPy, Matplotlib

------------------------------------------------------------------------

## 📁 Repository Structure (expected)

    /src
      segmentation.py
      matching.py
      homography.py
      evaluation.py
      utils.py

    /notebooks
      demo.ipynb

    /data
      sample_images/

    README.md

------------------------------------------------------------------------

## ▶ Usage

To run the pipeline on two images:

``` bash
python main.py --img1 path/to/img1.png --img2 path/to/img2.png
```

Outputs: - Estimated rotation (R) - Estimated translation direction
(t̂) - Visualization of matches - Homography reprojection plots

------------------------------------------------------------------------

## 📚 Reference Dataset

TUM RGB-D SLAM dataset (RGB only used).

------------------------------------------------------------------------

## 📄 Authors

-   **Anirudh (ME23B190)**\
-   **Harshavardhan (ME23B174)**\
    Course: **CS6350 Computer Vision**\
    Instructor: **Dr. Sukendu Das**, IIT Madras
