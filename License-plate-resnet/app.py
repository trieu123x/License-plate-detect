import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# PATCH: gradio_client has a known bug in certain versions where it crashes
# when parsing JSON schema with boolean additionalProperties.
# ----------------------------------------------------------------------------
try:
    import gradio_client.utils as _gc_utils

    _original_get_type = _gc_utils.get_type
    _original_json_schema_to_python_type = _gc_utils._json_schema_to_python_type

    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return _original_get_type(schema)

    def _patched_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _original_json_schema_to_python_type(schema, defs)

    _gc_utils.get_type = _patched_get_type
    _gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
    print("Đã patch gradio_client.utils (fix bug bool schema).")
except Exception as e:
    print(f"Không patch được gradio_client.utils: {e}")

import gradio as gr

# ----------------------------------------------------------------------------
# Setup and Model Initialization
# ----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# YOLO weight check
YOLO_WEIGHTS = os.path.join(SCRIPT_DIR, "best.pt")
if not os.path.exists(YOLO_WEIGHTS):
    YOLO_WEIGHTS = os.path.join(ROOT_DIR, "best.pt")
if not os.path.exists(YOLO_WEIGHTS):
    YOLO_WEIGHTS = os.path.join(ROOT_DIR, "detect", "train", "weights", "best.pt")

# ResNet50 weight check
RESNET_WEIGHTS = os.path.join(SCRIPT_DIR, "resnet50_finetuned.pth")
if not os.path.exists(RESNET_WEIGHTS):
    RESNET_WEIGHTS = os.path.join(SCRIPT_DIR, "resnet50_chars74k.pth")

print("Loading YOLO model...")
detector = YOLO(YOLO_WEIGHTS)

print("Loading ResNet model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(RESNET_WEIGHTS, map_location=device)
classes = checkpoint["classes"]

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

# Remove nn.DataParallel 'module.' prefix if present
state_dict = checkpoint["model_state_dict"]
clean_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    clean_state_dict[name] = v

model.load_state_dict(clean_state_dict)
model.to(device)
model.eval()
print(f"Models loaded successfully on device: {device}.")

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
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

# Valid characters for Vietnamese license plates (omit lowercase and I, O, Q, W)
VALID_CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZ"
valid_indices = [i for i, c in enumerate(classes) if map_char(c).upper() in VALID_CHARS]

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
    - grouped_rows: List of rows, where each row is a list of (x, y, w, h) bounding boxes.
    - thresh: Thresholded binary image.
    """
    plate = deskew_plate(plate)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Calculate block size dynamically based on plate height (approx. 25% of plate height, must be odd)
    plate_h, plate_w = plate.shape[:2]
    block_size = int(plate_h * 0.25)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(11, block_size)
    
    # Adaptive thresholding
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
            ix1 = max(x, fx)
            iy1 = max(y, fy)
            ix2 = min(x + w, fx + fw)
            iy2 = min(y + h, fy + fh)
            
            if ix2 > ix1 and iy2 > iy1:
                int_area = (ix2 - ix1) * (iy2 - iy1)
                box_area = w * h
                if int_area / float(box_area) > 0.7:
                    overlap = True
                    break
        if not overlap:
            filtered_boxes.append(box)

    # Sort boxes into rows (critical for 2-row plates)
    filtered_boxes = sorted(filtered_boxes, key=lambda b: b[1])
    
    rows = []
    for box in filtered_boxes:
        x, y, w, h = box
        assigned = False
        for row in rows:
            row_y_center = sum(r[1] + r[3]/2 for r in row) / len(row)
            box_y_center = y + h/2
            if abs(box_y_center - row_y_center) < (h * 0.5):
                row.append(box)
                assigned = True
                break
        if not assigned:
            rows.append([box])
            
    # Prune each row individually using median stats of that row
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
            
    # Sort rows from top-to-bottom
    pruned_boxes = sorted(pruned_boxes, key=lambda r: sum(b[1] for b in r) / len(r))
    
    # Sort each row from left-to-right
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

# ----------------------------------------------------------------------------
# Core Processing Pipeline
# ----------------------------------------------------------------------------
def detect_and_read_plate(image, conf, iou, pad_ratio, char_pad_ratio, invert_char, correct_syntax):
    """
    Inputs:
    - image: Input image (numpy array, RGB)
    - conf, iou: YOLO thresholds
    - pad_ratio: License plate padding
    - char_pad_ratio: Character crop padding
    - invert_char: Whether to invert colors (black text on white bg)
    - correct_syntax: Whether to enforce Vietnamese plate syntax rules
    
    Outputs:
    - output_image: Main image with drawn boxes
    - plate_crop: Cropped plate image
    - thresh_crop: Thresholded plate image
    - chars_gallery: List of segmented characters
    - output_text: Recognized plate text summary
    - html_plate: Stylized HTML representation
    """
    if image is None:
        return None, None, None, [], "Chưa có ảnh đầu vào.", ""

    # Gradio is RGB, OpenCV is BGR
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_display = img_bgr.copy()
    
    # YOLO inference
    results = detector(img_bgr, conf=conf, iou=iou, verbose=False)
    
    h_img, w_img = img_bgr.shape[:2]
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        return img_rgb, None, None, [], "Không phát hiện được biển số nào.", ""
        
    # Transform for ResNet50
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # We will return the first plate details for the side view panel
    first_plate_crop = None
    first_plate_thresh = None
    first_plate_chars = []
    
    plates_text_summary = []
    html_plates_list = []
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # Add padding to plate box
        pad_x = int((x2 - x1) * pad_ratio)
        pad_y = int((y2 - y1) * pad_ratio)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w_img, x2 + pad_x)
        y2 = min(h_img, y2 + pad_y)
        
        plate = img_bgr[y1:y2, x1:x2]
        if plate.size == 0:
            continue
            
        # Segment characters
        grouped_rows, thresh = segment_characters(plate)
        
        # Keep details of the first plate detected to show in the UI panel
        if idx == 0:
            first_plate_crop = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
            first_plate_thresh = thresh.copy()
            
        plate_text_rows = []
        total_chars_in_plate = 0
        total_rows = len(grouped_rows)
        
        for row_idx, row in enumerate(grouped_rows):
            row_text = ""
            row_len = len(row)
            for char_idx, char_box in enumerate(row):
                char_crop = crop_character_with_padding(thresh, char_box, char_pad_ratio)
                
                # Invert colors for model training alignment if requested (black text on white background)
                if invert_char:
                    char_crop = 255 - char_crop
                    
                # Pad to square and resize to 64x64 to preserve aspect ratio
                char_square = make_square_with_padding(char_crop, target_size=64, pad_color=255 if invert_char else 0)
                
                # Save first plate characters for the gallery
                if idx == 0:
                    first_plate_chars.append(char_square)
                    
                char_pil = Image.fromarray(char_square)
                tensor = transform(char_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(tensor)
                    
                # Mask invalid logits
                masked_pred = pred.clone()
                if correct_syntax:
                    allowed_chars = get_allowed_chars(row_idx, char_idx, total_rows, row_len)
                    allowed_indices = [i for i, c in enumerate(classes) if map_char(c).upper() in allowed_chars]
                    for idx_c in range(len(classes)):
                        if idx_c not in allowed_indices:
                            masked_pred[0, idx_c] = -float('inf')
                else:
                    for idx_c in range(len(classes)):
                        if idx_c not in valid_indices:
                            masked_pred[0, idx_c] = -float('inf')
                        
                cls_idx = masked_pred.argmax(1).item()
                char_val = map_char(classes[cls_idx]).upper()
                row_text += char_val
                total_chars_in_plate += 1
                
            plate_text_rows.append(row_text)
            
        # Construct final recognized plate text
        # Format rows nicely (e.g. Row1 \n Row2)
        plate_text = "".join(plate_text_rows)
        plates_text_summary.append(f"Biển số {idx+1}: {'-'.join(plate_text_rows)}")
        
        # HTML design representing a premium physical plate
        # Standard white plate with black border
        is_two_row = len(plate_text_rows) > 1
        if is_two_row:
            # Row 1 is usually short (like 30F), Row 2 is digits (like 123.45)
            r1 = plate_text_rows[0]
            r2 = plate_text_rows[1]
            # Formats bottom row: 12345 -> 123.45
            if len(r2) == 5:
                r2_formatted = f"{r2[:3]}.{r2[3:]}"
            else:
                r2_formatted = r2
                
            html_plate = f"""
            <div style="display: flex; justify-content: center; margin: 10px 0;">
                <div style="
                    border: 3px solid #111; 
                    border-radius: 8px; 
                    background-color: #fff; 
                    color: #000; 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    font-size: 28px; 
                    font-weight: 800; 
                    padding: 8px 18px; 
                    text-align: center; 
                    box-shadow: 0 4px 10px rgba(0,0,0,0.15); 
                    line-height: 1.1;
                    width: 160px;
                    border-bottom: 5px solid #222;
                ">
                    <span style="letter-spacing: 2px;">{r1}</span><br>
                    <span style="letter-spacing: 1px; font-size: 30px;">{r2_formatted}</span>
                </div>
            </div>
            """
        else:
            # 1 row plate: e.g. 30F-123.45
            r1 = plate_text_rows[0] if plate_text_rows else ""
            if len(r1) > 4:
                # E.g. 30F12345 -> 30F-123.45
                # Try to insert dashes appropriately
                # Vietnamese format: XXY-XXXX or XXY-XXX.XX
                # Check if it starts with standard 3 characters (e.g. 30F)
                letters_part = r1[:3]
                digits_part = r1[3:]
                if len(digits_part) == 5:
                    digits_formatted = f"{digits_part[:3]}.{digits_part[3:]}"
                else:
                    digits_formatted = digits_part
                r1_formatted = f"{letters_part}-{digits_formatted}"
            else:
                r1_formatted = r1
                
            html_plate = f"""
            <div style="display: flex; justify-content: center; margin: 10px 0;">
                <div style="
                    border: 3px solid #111; 
                    border-radius: 8px; 
                    background-color: #fff; 
                    color: #000; 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    font-size: 30px; 
                    font-weight: 800; 
                    padding: 8px 24px; 
                    text-align: center; 
                    box-shadow: 0 4px 10px rgba(0,0,0,0.15); 
                    letter-spacing: 2px;
                    border-bottom: 5px solid #222;
                    display: inline-block;
                ">
                    {r1_formatted}
                </div>
            </div>
            """
        html_plates_list.append(html_plate)
        
        # Draw bounding boxes and text on display image
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 3)
        text_y = y1 - 10 if y1 - 10 > 15 else y1 + 30
        cv2.putText(
            img_display,
            plate_text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
    img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    summary_text = "\n".join(plates_text_summary)
    html_plate_combined = "\n".join(html_plates_list)
    
    return (
        img_display_rgb, 
        first_plate_crop, 
        first_plate_thresh, 
        first_plate_chars, 
        summary_text, 
        html_plate_combined
    )

# ----------------------------------------------------------------------------
# Gradio Premium UI Setup
# ----------------------------------------------------------------------------
custom_css = """
body {
    background-color: #0d1117;
}
.gradio-container {
    font-family: 'Outfit', 'Inter', sans-serif !important;
}
h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}
.primary-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}
.primary-btn:hover {
    background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
}
"""

with gr.Blocks(title="License Plate Recognition System (YOLO + ResNet)", css=custom_css) as demo:
    gr.HTML(
        """
        <div style="text-align: center; padding: 20px 0; background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%); border-radius: 12px; margin-bottom: 25px;">
            <h1 style="color: #60a5fa; margin: 0; font-size: 36px; letter-spacing: 1px;">🚗 License Plate Recognition System</h1>
            <p style="color: #94a3b8; font-size: 16px; margin: 8px 0 0 0;">Nhận diện biển số thông minh sử dụng YOLOv8 & ResNet-50</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            # Input Column
            input_image = gr.Image(label="📷 Upload Ảnh Xe", type="numpy")
            
            with gr.Accordion("⚙️ Tùy Chỉnh Tham Số Nhận Diện", open=True):
                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                        label="YOLO Confidence"
                    )
                    iou_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                        label="YOLO IoU"
                    )
                with gr.Row():
                    pad_slider = gr.Slider(
                        minimum=0.0, maximum=0.3, value=0.05, step=0.01,
                        label="Padding Biển Số"
                    )
                    char_pad_slider = gr.Slider(
                        minimum=0.0, maximum=0.3, value=0.15, step=0.01,
                        label="Padding Ký Tự"
                    )
                invert_checkbox = gr.Checkbox(
                    value=True,
                    label="Đảo màu ký tự (Chữ đen trên nền trắng - Khuyên dùng cho ResNet74k)"
                )
                correct_syntax_checkbox = gr.Checkbox(
                    value=True,
                    label="Áp dụng luật cấu trúc biển số Việt Nam (Tự động sửa chữ nhầm thành số và ngược lại)"
                )
                
            submit_btn = gr.Button("🔍 Bắt Đầu Nhận Diện", variant="primary", elem_classes="primary-btn")
            
        with gr.Column(scale=4):
            # Results Column
            with gr.Tabs():
                with gr.TabItem("📊 Kết Quả Chính"):
                    output_image = gr.Image(label="🖼️ Ảnh Kết Quả")
                    output_text = gr.Textbox(label="📝 Text Biển Số Đọc Được", lines=3)
                    
                    gr.HTML("<h3 style='margin-top: 15px;'>💳 Biển Số Mô Phỏng</h3>")
                    html_plate_output = gr.HTML()
                    
                with gr.TabItem("🔬 Chi Tiết Pipeline OCR (First Plate)"):
                    with gr.Row():
                        plate_crop_img = gr.Image(label="🔲 Vùng Biển Số Cắt (YOLO)", interactive=False)
                        thresh_crop_img = gr.Image(label="🏁 Ảnh Binarized (Phân Đoạn)", interactive=False)
                    
                    gr.HTML("<h3 style='margin-top: 15px;'>🔠 Danh Sách Ký Tự Phân Tách</h3>")
                    chars_gallery_output = gr.Gallery(
                        label="Ký tự cắt lẻ đưa vào ResNet50", 
                        columns=10, 
                        height="auto",
                        object_fit="contain"
                    )

    # Click action
    submit_btn.click(
        fn=detect_and_read_plate,
        inputs=[
            input_image, 
            conf_slider, 
            iou_slider, 
            pad_slider, 
            char_pad_slider, 
            invert_checkbox,
            correct_syntax_checkbox
        ],
        outputs=[
            output_image, 
            plate_crop_img, 
            thresh_crop_img, 
            chars_gallery_output, 
            output_text, 
            html_plate_output
        ],
    )
    
    # Examples
    example_dir = os.path.join(SCRIPT_DIR, "assets")
    if not os.path.exists(example_dir):
        example_dir = os.path.join(ROOT_DIR, "assets")
    example_files = []
    for base in ["car", "car1", "car2", "car3", "car4"]:
        for ext in [".jpg", ".png", ".jpeg"]:
            fpath = os.path.join(example_dir, base + ext)
            if os.path.exists(fpath):
                example_files.append(fpath)
                break
    
    if example_files:
        gr.Examples(
            examples=example_files,
            inputs=input_image,
            label="💡 Chọn ảnh mẫu để test nhanh",
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
