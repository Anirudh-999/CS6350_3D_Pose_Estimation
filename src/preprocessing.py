import cv2
import numpy as np
from custom_logging import info, warn
from ultralytics import YOLO

_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    YOLO = None

def enhance_contrast(img_rgb: np.ndarray) -> np.ndarray:
    
    try:
        yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])

        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    
    except Exception:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])

        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def detect_yolo(img_rgb: np.ndarray, weights_path = "yolov8l", conf=0.1, imgsz=608):

    if not _YOLO_AVAILABLE:
        warn("ultralytics YOLO not installed. Skipping detection.")
        return np.empty((0,4)), np.array([]), np.array([])

    try:
        model = YOLO(weights_path)
        res = model(img_rgb, imgsz=imgsz, conf=conf)[0]
        if getattr(res,'boxes',None) is None:
            return np.empty((0,4)), np.array([]), np.array([])
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        info(f"YOLO detected {len(boxes)} boxes")
        return boxes, cls, confs
    
    except Exception as e:
        warn(f"YOLO inference failed: {e}")
        return np.empty((0,4)), np.array([]), np.array([])