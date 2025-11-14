from matplotlib import pyplot as plt
import os
import logging
import cv2
LOGGING_ENABLED = False
OUTPUT_DIR = "./output"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def info(message: str):
    """
    info - log an informational message
    """
    if LOGGING_ENABLED:
        logging.info(message)

def warn(message: str):
    """
    warn - log a warning message
    """
    if LOGGING_ENABLED:
        logging.warning(message)

def err(message: str):
    """
    err - log an error message
    """
    if LOGGING_ENABLED:        
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
