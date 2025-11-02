import cv2
import numpy as np
import os
from ultralytics import YOLO
from matplotlib import pyplot as plt

def detect_yolo(img_rgb: np.ndarray, weights_path = "yolov8l.pt", conf=0.25, imgsz=640):
   
    try:
        model = YOLO(weights_path)
        res = model(img_rgb, imgsz=imgsz, conf=conf)[0]
        if getattr(res,'boxes',None) is None:
            return np.empty((0,4)), np.array([]), np.array([])
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        print(f"YOLO detected {len(boxes)} boxes")
        return boxes, cls, confs
    
    except Exception as e:
        print(f"YOLO inference failed: {e}")
        return np.empty((0,4)), np.array([]), np.array([])

img_path = '../images/img4.jpeg'

img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
boxes, cls, confs = detect_yolo(img_rgb)

img_vis = img_rgb.copy()
for (x1, y1, x2, y2), c, conf in zip(boxes, cls, confs):
    cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(img_vis, f"{c}:{conf:.2f}", (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# show output inline
plt.figure(figsize=(10, 6))
plt.imshow(img_vis)
plt.axis('off')
plt.title("YOLO Detection Output")
plt.show()