# CS6350: 3D Pose Estimation from Two Uncalibrated Views

## Overview
This repository implements a pipeline for estimating the relative 6-DoF pose between two uncalibrated RGB images of a near-planar object. The objective is to assess whether reliable pose change can be recovered without camera intrinsics, depth information, or multi-view constraints.

---

## Method Summary

### 1. Segmentation and ROI Extraction
The target object is isolated using FastSAM.  
A tight region of interest (ROI) is extracted to remove background features and improve feature matching quality.

### 2. Edge-Aware Feature Matching
Canny edges are detected on each ROI to identify reliable geometric boundaries.  
LoFTR provides dense feature correspondences, which are filtered to keep only edge-aligned pairs.  
Fallback strategies (raw LoFTR and SIFT) are used when edge matches are insufficient.

### 3. Homography Estimation
RANSAC is applied to the matched points to robustly estimate a planar homography mapping between the two views.

### 4. Homography Decomposition
The homography is decomposed into rotation and translation candidates using an assumed intrinsic matrix (uncalibrated).  
Invalid solutions are discarded using geometric consistency checks.

### 5. Evaluation
Ground-truth poses from the TUM RGB-D dataset are used **only for evaluation**, not during estimation.  
Metrics include:
- rotation error (degrees)
- translation-direction error
- translation-scale error
- reprojection RMS error

---

## Dataset
Experiments use only the RGB images from the TUM RGB-D dataset.  
Depth and camera intrinsics are intentionally excluded to maintain the uncalibrated monocular setting.



