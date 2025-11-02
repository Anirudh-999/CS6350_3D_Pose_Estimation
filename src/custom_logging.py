from matplotlib import pyplot as plt
import os
import logging
import cv2

# LEFT_IMG_PATH  = "images/img2.png"      # path to left image
# RIGHT_IMG_PATH = "images/img1.png"      # path to right image

LEFT_IMG_PATH  = "../images/img3.jpeg"      # path to left image
RIGHT_IMG_PATH = "../images/img4.jpeg"      # path to right image

CALIB_PATH     = "calib.txt"      # optional KITTI-style calib (P0/P1 or K)
YOLO_WEIGHTS   = "yolov8l.pt"     # optional (if ultralytics installed)
OUTPUT_DIR     = "./output"       # where we save visualizations
USE_GPU        = True             # preference flag

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def info(message: str):
    logging.info(message)

def warn(message: str):
    logging.warning(message)

def err(message: str):
    logging.error(message)

def show_and_save(img_rgb, title: str, fname: str, output_dir: str = OUTPUT_DIR):
    """
    Save an image to the specified output directory and log the action.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, fname)
    
    # Convert RGB to BGR before saving
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)
    info(f"Saved figure: {path}")
