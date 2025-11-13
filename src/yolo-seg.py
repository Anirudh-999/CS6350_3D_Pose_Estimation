from ultralytics import FastSAM
import cv2
import numpy as np  # Import numpy for calculations

# --- 1. Load the FastSAM Model ---
model = FastSAM('FastSAM-s.pt')  # Use FastSAM-l (large) or FastSAM-s (small)

# Define your image path
image_path = r'output\01_left.png'
# Or use a sample image
# image_path = 'bus.jpg'

img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image from {image_path}")
else:
    # --- 2. Run Inference ---
    results = model(img, device='cpu', retina_masks=True, conf=0.4)

    # Get the first (and only) result
    result = results[0]

    # --- 3. Find the Second Largest Segment ---
    if result.masks:
        print(f"\nFound {len(result.masks)} total segments. Finding second largest...")

        largest_area = 0
        largest_index = -1
        second_largest_area = 0
        second_largest_index = -1
        
        # Loop through each mask to find the one with the largest area
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
        
        # --- 4. Manually Draw Mask and TIGHT Bounding Box ---
        # Check if we found a valid second segment
        if second_largest_index != -1:
            print(f"Displaying second largest segment (Index: {second_largest_index}) with tight box.")

            # Get the mask data for the chosen segment
            mask_data = result.masks[second_largest_index]

            # 4a. Get the raw pixel mask (it's low-res, e.g., 160x160)
            # .data[0] gives us the [H, W] mask tensor
            low_res_mask = mask_data.data[0].cpu().numpy() # Move to CPU and convert to numpy
            
            # 4b. Resize the mask to the original image's size
            orig_h, orig_w = img.shape[:2]
            full_res_mask = cv2.resize(low_res_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # 4c. Threshold the resized mask to get a binary (0 or 255) image
            # This gives us the precise pixel-level segmentation
            _, binary_mask = cv2.threshold(full_res_mask, 0.5, 255, cv2.THRESH_BINARY)
            binary_mask = binary_mask.astype(np.uint8) # Convert to uint8 for findContours

            # 4d. Find contours from this precise binary mask
            # This is the key step to "remove outliers" and get the true shape
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 4e. Find the single largest contour (in case the mask has small, stray blobs)
                main_contour = max(contours, key=cv2.contourArea)

                # --- Draw the results manually ---
                output_image = img.copy()
                color = (0, 255, 0) # Green
                alpha = 0.4 # Transparency

                # 4f. Draw the filled-in mask
                overlay = output_image.copy()
                cv2.drawContours(overlay, [main_contour], -1, color, -1) # -1 thickness = filled
                
                # Blend the mask with the original image
                output_image = cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0)

                # 4g. Calculate the TIGHT bounding box from the contour
                x, y, w, h = cv2.boundingRect(main_contour)
                
                # 4h. Draw the TIGHT bounding box
                cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2) # 2px thickness

                # Display the image
                cv2.imshow("FastSAM - Second Largest (Tight Box)", output_image)
            else:
                print("Could not find contours in the resized mask.")
                cv2.imshow("FastSAM - No Contours Found", img)

        else:
            print("Could not find a valid second largest segment (need at least 2 segments).")
            cv2.imshow("FastSAM - No Second Segment Found", img)
            
    else:
        print("\nNo segments found in the image.")
        cv2.imshow("FastSAM - No Segments Found", img)


    # --- 5. Wait for key press ---
    print("Displaying FastSAM image. Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()