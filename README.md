# CS6350: 3D Pose Estimation from Two Uncalibrated Views

## Overview
This repository implements a pipeline for estimating the relative 6-DoF pose between two uncalibrated RGB images of a near-planar object. The objective is to assess whether reliable pose change can be recovered without camera intrinsics, depth information, or multi-view constraints.

---

## Method Summary

### 1. Segmentation and ROI Extraction
The target object is isolated using FastSAM.  
A region of interest (ROI) is extracted to remove background features and improve feature matching quality.

### 2. Edge-Aware Feature Matching
LoFTR provides dense feature correspondences, which are filtered to keep only edge-aligned pairs.  
Canny edges are detected on each ROI to identify reliable geometric boundaries.  

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

## 6. Prerequisites
* **OS:** Windows, macOS, or Linux.
* **Python:** Version 3.8 or newer.
* **Hardware:** A generic CPU is sufficient.

## 3. Installation Guide

Follow these steps to set up your environment cleanly.

### Folder Structure
Ensure your folders are organized exactly like this. The code relies on specific relative paths (e.g., `../dataset/`).

```text
project_root/
│
├── requirements.txt       # The dependency list provided above
├── README.md              # This file
│
├── src/                   # Source code folder
│   ├── data_handling.py   # <--- Run this file to start the program
│   ├── main.py
│   ├── preprocessing.py
│   ├── input_output.py
│   ├── epipolar_geometry.py
│   ├── feature_extraction_maping.py
│   ├── midas_estimate.py
│   ├── triangulation_estimation.py
│   ├── custom_logging.py
│   └── output/            # Results will be saved here automatically
│
└── dataset/               # Data folder (one level up from src)
    ├── rgb.txt            # Timestamp-to-filename list (TUM format)
    ├── groundtruth.txt    # Ground truth poses (TUM format)
    └── rgb/               # Folder containing the actual .png images
        ├── image1.png
        └── ...

### Step 1: Get the Code

**Option A: Clone via Git**
Open your terminal (Command Prompt, PowerShell, or Terminal) and run:
```bash
git clone https://github.com/ANIRUDH-999/CS6350.git
cd YOUR_REPO_NAME
cd project_root
python -m venv venv
.\venv\Scripts\activate

## Dataset
Experiments use only the RGB images from the TUM RGB-D dataset.  
Depth and camera intrinsics are intentionally excluded to maintain the uncalibrated monocular setting.
https://cvg.cit.tum.de/data/datasets/rgbd-dataset
https://cvg.cit.tum.de/data/datasets/rgbd-dataset

