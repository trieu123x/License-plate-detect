import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

def map_char(class_name):
    """
    Maps class name to the actual character.
    If it's already a single character, returns it directly.
    """
    if len(class_name) == 1:
        return class_name
    if class_name.startswith("Sample"):
        try:
            num = int(class_name.replace("Sample", ""))
            if 1 <= num <= 10:
                return chr(ord('0') + num - 1)
            elif 11 <= num <= 36:
                return chr(ord('A') + num - 11)
            elif 37 <= num <= 62:
                return chr(ord('a') + num - 37)
        except ValueError:
            pass
    return class_name

def deskew_plate(plate):
    """
    Detects the skew angle of the license plate using line fitting on character centroids
    and rotates it to align horizontally.
    """
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    plate_h, plate_w = plate.shape[:2]
    block_size = int(plate_h * 0.25)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(11, block_size)
    
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        9
    )
    
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    points = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if (plate_h * 0.2 < h < plate_h * 0.95) and (w < plate_w * 0.4):
            if w > 2 and h > 10 and (0.18 < aspect_ratio < 0.9):
                cx = x + w / 2.0
                cy = y + h / 2.0
                points.append((cx, cy))
                
    if len(points) < 3:
        return plate
        
    pts = np.array(points, dtype=np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    
    angle = np.degrees(np.arctan2(vy[0], vx[0]))
    
    if abs(angle) < 0.5 or abs(angle) > 25.0:
        return plate
        
    print(f"Detected plate skew angle: {angle:.2f} degrees. Rotating.")
    center = (plate_w // 2, plate_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    bg_color = int(np.median([gray[0,0], gray[0, plate_w-1], gray[plate_h-1, 0], gray[plate_h-1, plate_w-1]]))
    border_val = (bg_color, bg_color, bg_color)
    
    rotated = cv2.warpAffine(plate, M, (plate_w, plate_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return rotated

def segment_characters(plate):
    """
    Segment characters from the license plate image using adaptive thresholding.
    Returns:
    - sorted_boxes: List of (x, y, w, h) bounding boxes sorted correctly.
    - thresh: Thresholded binary image.
    """
    plate = deskew_plate(plate)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Calculate block size dynamically based on plate height (approx. 25% of plate height, must be odd)
    plate_h, plate_w = plate.shape[:2]
    block_size = int(plate_h * 0.25)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(11, block_size)
    
    # Use Adaptive Thresholding with dynamic block size
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        9
    )
    
    # Clear the borders of the thresholded image to disconnect characters touching the edge
    border_thickness = max(2, int(min(plate_h, plate_w) * 0.05))
    cv2.rectangle(thresh, (0, 0), (plate_w - 1, plate_h - 1), 0, border_thickness)
    
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    boxes = []
    
    # Filter contours based on dimensions relative to plate size
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        
        # Heuristics for a character:
        # 1. Height should be between 20% and 95% of the plate height
        # 2. Width should be less than 40% of the plate width
        # 3. Width should be at least 3 pixels, height at least 10 pixels
        # 4. Aspect ratio should be reasonable (not too wide, not too thin)
        if (plate_h * 0.2 < h < plate_h * 0.95) and (w < plate_w * 0.4):
            if w > 2 and h > 10 and (0.18 < aspect_ratio < 0.9):
                boxes.append((x, y, w, h))
                
    # Remove nested or highly overlapping bounding boxes
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    filtered_boxes = []
    for box in boxes:
        x, y, w, h = box
        overlap = False
        for f_box in filtered_boxes:
            fx, fy, fw, fh = f_box
            # Calculate intersection
            ix1 = max(x, fx)
            iy1 = max(y, fy)
            ix2 = min(x + w, fx + fw)
            iy2 = min(y + h, fy + fh)
            
            if ix2 > ix1 and iy2 > iy1:
                int_area = (ix2 - ix1) * (iy2 - iy1)
                box_area = w * h
                # If intersection area is > 70% of the current box, discard it
                if int_area / float(box_area) > 0.7:
                    overlap = True
                    break
        if not overlap:
            filtered_boxes.append(box)

    # Sort boxes into rows (critical for 2-row plates)
    # 1. Sort all boxes top-to-bottom first
    filtered_boxes = sorted(filtered_boxes, key=lambda b: b[1])
    
    # 2. Group into rows
    rows = []
    for box in filtered_boxes:
        x, y, w, h = box
        assigned = False
        for row in rows:
            # Calculate average Y center of current row
            row_y_center = sum(r[1] + r[3]/2 for r in row) / len(row)
            box_y_center = y + h/2
            # If the box center is vertically close to the row center, add it to this row
            # We use a threshold of 50% of the box's height
            if abs(box_y_center - row_y_center) < (h * 0.5):
                row.append(box)
                assigned = True
                break
        if not assigned:
            rows.append([box])
            
    # 3. Prune each row individually using median stats of that row
    pruned_boxes = []
    for row in rows:
        if not row:
            continue
        row_heights = [b[3] for b in row]
        row_median_h = np.median(row_heights)
        row_centers_y = [b[1] + b[3]/2 for b in row]
        row_median_y_center = np.median(row_centers_y)
        
        pruned_row = []
        for box in row:
            x, y, w, h = box
            y_center = y + h/2
            aspect_ratio = w / float(h)
            
            # Keep character if height, alignment, and aspect ratio are within range of row median
            if (0.7 * row_median_h <= h <= 1.3 * row_median_h) and \
               (abs(y_center - row_median_y_center) < row_median_h * 0.25) and \
               (0.18 <= aspect_ratio <= 0.9):
                pruned_row.append(box)
        if pruned_row:
            pruned_boxes.append(pruned_row)
            
    # 4. Sort rows from top-to-bottom (by average y-coordinate of row)
    # 5. Sort each row from left-to-right (by x-coordinate)
    pruned_boxes = sorted(pruned_boxes, key=lambda r: sum(b[1] for b in r) / len(r))
    sorted_rows = []
    for row in pruned_boxes:
        sorted_row = sorted(row, key=lambda b: b[0])
        sorted_rows.append(sorted_row)
        
    return sorted_rows, thresh

def clean_character_component(char_crop_raw):
    """
    Keep only the largest connected component in the exact crop, 
    setting everything else (noise, borders) to 0 (background).
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(char_crop_raw, connectivity=8)
    if num_labels <= 1:
        return char_crop_raw
        
    largest_label = 1
    largest_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i
            
    mask = (labels == largest_label).astype(np.uint8) * 255
    return mask

def crop_character_with_padding(thresh, box, pad_ratio=0.15):
    """
    Crops a character from thresholded image with a relative padding,
    using Connected Components Analysis to filter noise and constant border padding.
    """
    cx, cy, cw, ch = box
    exact_crop = thresh[cy:cy+ch, cx:cx+cw].copy()
    clean_crop = clean_character_component(exact_crop)
    
    pad_h = int(ch * pad_ratio)
    pad_w = int(cw * pad_ratio)
    
    padded_char = cv2.copyMakeBorder(
        clean_crop, pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_CONSTANT, value=0
    )
    return padded_char

def get_allowed_chars(row_idx, char_idx, total_rows, row_len):
    """
    Returns standard allowed characters for each position in Vietnamese plates
    to prevent letter/number confusion.
    """
    # 2-Row Plate (e.g. Motorcycle: 59-T1 / 172.01)
    if total_rows == 2:
        if row_idx == 0:
            if row_len == 4:
                if char_idx in [0, 1]:
                    return "0123456789"
                elif char_idx == 2:
                    return "ABCDEFGHKLMNPRSTUVXYZ"
                elif char_idx == 3:
                    return "0123456789"
            elif row_len == 3:
                if char_idx in [0, 1]:
                    return "0123456789"
                elif char_idx == 2:
                    return "ABCDEFGHKLMNPRSTUVXYZ"
        elif row_idx == 1:
            return "0123456789"
            
    # 1-Row Plate (e.g. Car: 30F-123.45)
    elif total_rows == 1:
        if char_idx in [0, 1]:
            return "0123456789"
        elif char_idx == 2:
            return "ABCDEFGHKLMNPRSTUVXYZ"
        elif char_idx >= 3:
            return "0123456789"
            
    return "0123456789ABCDEFGHKLMNPRSTUVXYZ"

def make_square_with_padding(img, target_size=64, pad_color=255):
    """
    Pads the image to be a square and resizes it to target_size without distortion.
    """
    h, w = img.shape[:2]
    if h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        squared_img = cv2.copyMakeBorder(
            img, 0, 0, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=pad_color
        )
    elif w > h:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        squared_img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, 0, 0, 
            cv2.BORDER_CONSTANT, value=pad_color
        )
    else:
        squared_img = img.copy()
        
    # Choose interpolation based on whether we are upscaling or downscaling
    h_sq, w_sq = squared_img.shape[:2]
    if h_sq < target_size:
        interp = cv2.INTER_LINEAR  # Better for upscaling (enlarging)
    else:
        interp = cv2.INTER_AREA    # Better for downscaling (shrinking)
        
    return cv2.resize(squared_img, (target_size, target_size), interpolation=interp)

def main():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # Paths to models
    yolo_weights = os.path.join(root_dir, "best.pt")
    if not os.path.exists(yolo_weights):
        yolo_weights = os.path.join(root_dir, "detect", "train", "weights", "best.pt")
        
    resnet_weights = os.path.join(script_dir, "resnet50_finetuned.pth")
    if not os.path.exists(resnet_weights):
        resnet_weights = os.path.join(script_dir, "resnet50_chars74k.pth")
    
    print(f"Loading YOLO from: {yolo_weights}")
    print(f"Loading ResNet from: {resnet_weights}")
    
    # Load YOLO
    detector = YOLO(yolo_weights)
    
    # Load ResNet50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint = torch.load(resnet_weights, map_location=device)
    classes = checkpoint["classes"]
    
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    
    # Clean up state_dict keys (remove 'module.' prefix if present)
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        clean_state_dict[name] = v
        
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    
    # Valid characters for Vietnamese license plates (omit lowercase and I, O, Q, W)
    VALID_CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZ"
    valid_indices = [i for i, c in enumerate(classes) if map_char(c).upper() in VALID_CHARS]
    
    # Load a test image
    test_img_path = os.path.join(root_dir, "assets", "car4.jpg")
    if not os.path.exists(test_img_path):
        # Fallback to any image in assets
        assets_dir = os.path.join(root_dir, "assets")
        if os.path.exists(assets_dir):
            files = [f for f in os.listdir(assets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                test_img_path = os.path.join(assets_dir, files[0])
                
    if not os.path.exists(test_img_path):
        print(f"Error: No test image found at {test_img_path}")
        return
        
    print(f"Reading image: {test_img_path}")
    img = cv2.imread(test_img_path)
    img_display = img.copy()
    
    # Run YOLO detection
    results = detector(img, conf=0.5, iou=0.5)
    
    # Transform for ResNet50
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    
    h_img, w_img = img.shape[:2]
    
    for idx, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        
        # Add 5% padding around the plate
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.05)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w_img, x2 + pad_x)
        y2 = min(h_img, y2 + pad_y)
        
        plate = img[y1:y2, x1:x2]
        if plate.size == 0:
            continue
            
        print(f"\n--- Processing License Plate {idx+1} ---")
        
        # Segment characters
        grouped_rows, thresh = segment_characters(plate)
        total_chars = sum(len(row) for row in grouped_rows)
        print(f"Found {total_chars} character contours.")
        
        plate_text = ""
        
        # Create output directory for cropped characters
        cropped_chars_dir = os.path.join(script_dir, "cropped_chars")
        os.makedirs(cropped_chars_dir, exist_ok=True)
        print(f"Saving cropped characters to: {cropped_chars_dir}")
        
        # Classify each character
        char_counter = 0
        total_rows = len(grouped_rows)
        for row_idx, row in enumerate(grouped_rows):
            row_len = len(row)
            for char_idx, (cx, cy, cw, ch) in enumerate(row):
                # Crop character with 15% padding
                char_crop = crop_character_with_padding(thresh, (cx, cy, cw, ch), pad_ratio=0.15)
                
                # Save raw cropped character (white on black)
                raw_crop = thresh[cy:cy+ch, cx:cx+cw]
                cv2.imwrite(os.path.join(cropped_chars_dir, f"char_raw_{char_counter}.png"), raw_crop)
                
                # Save processed character (black on white, with 15% padding)
                char_crop_inverted = 255 - char_crop
                # Pad to square and resize to 64x64
                char_square = make_square_with_padding(char_crop_inverted, target_size=64, pad_color=255)
                cv2.imwrite(os.path.join(cropped_chars_dir, f"char_processed_{char_counter}.png"), char_square)
                
                char_pil = Image.fromarray(char_square)
                
                # Predict
                tensor = transform(char_pil)
                tensor = tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(tensor)
                    
                # Mask invalid logits using position-based rules
                masked_pred = pred.clone()
                allowed_chars = get_allowed_chars(row_idx, char_idx, total_rows, row_len)
                allowed_indices = [i for i, c in enumerate(classes) if map_char(c).upper() in allowed_chars]
                for idx_c in range(len(classes)):
                    if idx_c not in allowed_indices:
                        masked_pred[0, idx_c] = -float('inf')
                        
                cls = masked_pred.argmax(1).item()
                class_name = classes[cls]
                char_val = map_char(class_name).upper()
                plate_text += char_val
                char_counter += 1
                
        # Convert lowercase to uppercase (standard for Vietnamese plates)
        plate_text = plate_text.upper()
        print(f"Recognized Plate Text: {plate_text}")
        
        # Draw bounding boxes and text
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            img_display,
            plate_text,
            (x1, max(15, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
    # Save the output image
    output_path = os.path.join(script_dir, "output_result.jpg")
    cv2.imwrite(output_path, img_display)
    print(f"\nSuccess! Result saved to {output_path}")

if __name__ == "__main__":
    main()
