import cv2
import numpy as np
from custom_logging import info, warn, err, show_and_save

# Import both YOLO (for detect_yolo) and FastSAM (for segment_and_get_largest_box)
from ultralytics import YOLO, FastSAM

_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO, FastSAM
    _YOLO_AVAILABLE = True
    _FASTSAM_AVAILABLE = True
except Exception:
    YOLO = None
    FastSAM = None

def enhance_contrast(img_rgb: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE in YUV or LAB color space.

    returns enhanced RGB image.
    """
    
    try:
        yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])

        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    
    except Exception:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])

        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def segment_and_get_largest_box(img_rgb, model_path="yolov8l-seg.pt", save_mask_path=None):

    """
    Runs FastSAM segmentation on an image, finds the 2nd largest segment mask,
    and returns its tight bounding box AND the masked image.
    (Note: Ignores model_path to use FastSAM as requested).
    
    Returns: (boxes, cls_ids, confs, masked_image)
    """
    
    if not _YOLO_AVAILABLE or FastSAM is None:
        warn("FastSAM (ultralytics) not installed, skipping segmentation.")
        # --- MODIFIED: Return 4 values ---
        return np.empty((0, 4)), [None], [0.0], None

    try:
        # 1. Load Model
        info("Loading FastSAM-s.pt model for segmentation...")
        model = FastSAM('FastSAM-s.pt')
        
        # 2. Run Inference
        results = model(img_rgb, device='cpu', retina_masks=True, conf=0.4, verbose=False)
        result = results[0]

        if result.masks is None or len(result.masks) == 0:
            warn("No segments found in image.")
            # --- MODIFIED: Return 4 values ---
            return np.empty((0, 4)), [None], [0.0], None

        # 3. Find the Second Largest Segment
        info(f"Found {len(result.masks)} total segments. Finding second largest...")
        largest_area = 0
        largest_index = -1
        second_largest_area = 0
        second_largest_index = -2
        
        for i, mask_data in enumerate(result.masks):
            polygon_points = mask_data.xy[0]
            area = cv2.contourArea(np.int32(polygon_points))

            if area > largest_area:
                second_largest_area = largest_area
                second_largest_index = largest_index
                largest_area = area
                largest_index = i
            elif area > second_largest_area:
                second_largest_area = area
                second_largest_index = i

        # 4. Get the mask and calculate the tight box
        if second_largest_index == -1:
            if largest_index != -1:
                 info("Only one segment found, using the largest as fallback.")
                 second_largest_index = largest_index # Fallback to largest
            else:
                warn("Could not find any valid segments.")
                # --- MODIFIED: Return 4 values ---
                return np.empty((0, 4)), [None], [0.0], None

        info(f"Using segment index {second_largest_index} (Area: {second_largest_area})")

        mask_data = result.masks[second_largest_index]

        # Get the raw pixel mask
        low_res_mask = mask_data.data[0].cpu().numpy()
        
        # Resize the mask to the original image's size
        orig_h, orig_w = img_rgb.shape[:2]
        full_res_mask = cv2.resize(low_res_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Threshold to get a binary mask
        _, binary_mask = cv2.threshold(full_res_mask, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask_uint8 = binary_mask.astype(np.uint8)

        # MODIFIED: Create masked image 
        # Create the masked image: object on black background
        masked_image = cv2.bitwise_and(img_rgb, img_rgb, mask=binary_mask_uint8)

        # Find contours from this precise binary mask
        contours, _ = cv2.findContours(binary_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            warn("Could not find contours in the resized mask.")
            # MODIFIED: Return 4 values
            return np.empty((0, 4)), [None], [0.0], None

        # Find the single largest contour
        main_contour = max(contours, key=cv2.contourArea)

        # Calculate the TIGHT bounding box from the contour
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Save the mask visualization
        if save_mask_path:
            try:
                # Use the 'masked_image' we already created
                show_and_save(masked_image, title=f"Used Segment Mask (FastSAM)", fname=save_mask_path[10:], output_dir=save_mask_path)
            except Exception as e:
                warn(f"Could not save segmentation mask: {e}")
        
        # Format as [x1, y1, x2, y2]
        box = np.array([[x, y, x + w, y + h]])
        
        # Return the masked_image as the 4th value
        return box, [None], [1.0], masked_image

    except Exception as e:
        err(f"FastSAM segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return np.empty((0, 4)), [None], [0.0], None

def detect_yolo(img_rgb: np.ndarray, weights_path = "yolov8l", conf=0.1, imgsz=608):

    """
    Runs YOLO object detection on an image.

    Returns: boxes (Nx4), class_ids (N,), confidences (N,)
    """

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

def refine_matches_photometric(img1, img2, pts1):
    """
    Refine matches using Lucas-Kanade optical flow for subpixel accuracy.

    return pts1_refined, pts2_refined
    """

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    pts1 = np.array(pts1, dtype=np.float32).reshape(-1, 1, 2)
    pts2, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1, None, 
                                             winSize=(15, 15),
                                             maxLevel=3,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    # Filter points with valid status
    valid = st.ravel() == 1
    pts1_refined = pts1[valid].reshape(-1, 2)
    pts2_refined = pts2[valid].reshape(-1, 2)

    return pts1_refined, pts2_refined