## 3D Pose Change Estimation (Uncalibrated Two-View)

This repository implements a pipeline for estimating the **6-DoF relative pose change** of a *near-planar object* using only **two uncalibrated RGB images**.

This project was developed as part of the **CS6350 -- Computer Vision** coursework at IIT Madras.

---

### Key Idea and Methodology

Recovering 3D pose change from two uncalibrated images is inherently challenging due to the absence of camera intrinsic parameters, the non-recoverability of translation scale, and the failure of standard feature matching on near-planar, low-texture objects.

The system addresses these limitations by employing a strategy based on **segmentation**, **edge-aware feature matching**, and **homography decomposition**.

---

### Pipeline Overview

The complete estimation process involves five sequential steps:

1.  **Segmentation (FastSAM)**
    * Extracts the precise object mask.
    * Crops a tight Region of Interest (ROI) around the object.
    * Effectively removes background clutter.
2.  **Edge-Aware Matching**
    * Computes image edges using the Canny detector.
    * Restricts LoFTR matches to lie on computed edge pixels.
    * Filters SIFT matches using the edge map for increased robustness.
    * Includes a fallback to raw LoFTR or SIFT matching if edge matching yields insufficient points.
3.  **Homography Estimation**
    * Performs RANSAC-based estimation of the homography matrix ($H$).
    * Approximates the camera intrinsic parameters ($K$) as:
        $$f_x = f_y = \max(W, H)$$
        $$c_x = W/2, c_y = H/2$$
        where $W$ and $H$ are the image width and height.
4.  **Homography Decomposition**
    * Extracts candidate triplets of Rotation ($R$), Translation ($t$), and Plane Normal ($n$).
    * Filters physically invalid solutions.
    * Selects the optimal (R, t, n) triplet using geometric heuristics.
5.  **Evaluation (Using TUM RGB-D Ground Truth)**
    * Quantifies Rotation error.
    * Quantifies Translation direction error.
    * Calculates the Reprojection Root Mean Square (RMS) error.

---

### Results Summary

| Motion Type | Pairs | Rotation Error (MAE) | Translation Direction Error | Key Observations |
| :--- | :--- | :--- | :--- | :--- |
| **Small-frame** | $\approx 1100$ | $\approx 0.6^{\circ}$ | High (Inherent limitation of homography) | High accuracy for Roll and Pitch rotations. Yaw rotation is unstable due to planar geometry. Scale is fundamentally unrecoverable. |
| **Large-frame** | $\approx 1000$ | Rises to $\approx 9^{\circ}$ | N/A | Failures correlate with ambiguous homography decomposition selecting an incorrect branch. Reprojection RMS remains low, confirming correct feature matching despite decomposition failure. |

---

### Insights and Limitations

#### Strengths

* Robust and stable homography estimation achieved through edge-aware feature matching.
* Accurate Roll and Pitch rotation recovery for near-planar objects.

#### Limitations

* Translation is only recovered up to an arbitrary scale factor.
* Yaw rotation is poorly constrained and observed in planar configurations.
* Large viewpoint changes lead to ambiguity in homography decomposition results.

---

### Requirements

* Python 3.8+
* OpenCV
* PyTorch
* LoFTR model files
* FastSAM
* NumPy, Matplotlib

---

### Repository Structure
