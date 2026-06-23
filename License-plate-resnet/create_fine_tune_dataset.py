import os
import cv2
import numpy as np
import sys
from tqdm import tqdm

# Add parent directory to path to import test_cli
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_cli import segment_characters, crop_character_with_padding, make_square_with_padding
from ultralytics import YOLO
from paddleocr import PaddleOCR

def clean_ocr_text(text):
    VALID_CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZ"
    cleaned = ""
    for char in text.upper():
        if char in VALID_CHARS:
            cleaned += char
    return cleaned

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # Target dataset folder
    dataset_dir = os.path.join(script_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Initialize YOLO detector
    yolo_weights = os.path.join(root_dir, "best.pt")
    if not os.path.exists(yolo_weights):
        yolo_weights = os.path.join(root_dir, "detect", "train", "weights", "best.pt")
        
    print(f"Loading YOLO from: {yolo_weights}")
    detector = YOLO(yolo_weights)
    
    # Initialize PaddleOCR
    print("Loading PaddleOCR...")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en"
    )
    
    # Classes setup and resuming
    VALID_CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZ"
    class_counts = {}
    processed_images = set()
    for char in VALID_CHARS:
        char_dir = os.path.join(dataset_dir, char)
        os.makedirs(char_dir, exist_ok=True)
        class_counts[char] = len(os.listdir(char_dir))
        for file in os.listdir(char_dir):
            if "_plate0_char" in file:
                img_basename = file.split("_plate0_char")[0]
                processed_images.add(img_basename)
                
    print(f"Resuming with {len(processed_images)} already processed images.")
    print("Current counts:")
    for char, count in sorted(class_counts.items()):
        if count > 0:
            print(f"  Class '{char}': {count} images")
            
    # Gather all source image paths from Roboflow dataset splits
    splits = ["train", "valid", "test"]
    image_paths = []
    
    for split in splits:
        split_img_dir = os.path.join(root_dir, "Vietnam license-plate.v1i.yolov8", split, "images")
        if os.path.exists(split_img_dir):
            for file in os.listdir(split_img_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_basename = os.path.splitext(file)[0]
                    if img_basename not in processed_images:
                        image_paths.append((split, os.path.join(split_img_dir, file)))
                    
    print(f"Remaining images to process: {len(image_paths)}")
    
    saved_total = sum(class_counts.values())
    
    for split, img_path in tqdm(image_paths, desc="Processing images"):
        # Check if we have collected 100 for all classes
        if all(count >= 100 for count in class_counts.values()):
            print("Reached target of 100 crops for all classes. Stopping.")
            break
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h_img, w_img = img.shape[:2]
        
        # Run YOLO plate detection
        results = detector(img, conf=0.5, iou=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) == 0:
            continue
            
        # Process the first detected plate
        box = boxes[0]
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
            
        # Run PaddleOCR on plate
        try:
            ocr_res = ocr.predict(plate)
            if not ocr_res or len(ocr_res) == 0:
                continue
            
            data = ocr_res[0]
            if not data or 'rec_texts' not in data or not data['rec_texts']:
                continue
                
            raw_text = " ".join(data['rec_texts'])
            cleaned_text = clean_ocr_text(raw_text)
            if len(cleaned_text) == 0:
                continue
        except Exception as e:
            continue
            
        # Segment characters using our OpenCV logic
        char_boxes, thresh = segment_characters(plate)
        
        # 1-to-1 match verification
        if len(char_boxes) == len(cleaned_text):
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            
            for idx, (cx, cy, cw, ch) in enumerate(char_boxes):
                char_label = cleaned_text[idx]
                
                # Skip if we already have 100 images for this class
                if class_counts[char_label] >= 100:
                    continue
                    
                # Crop character with 15% padding
                char_crop = crop_character_with_padding(thresh, (cx, cy, cw, ch), pad_ratio=0.15)
                # Invert (black text on white bg)
                char_crop_inverted = 255 - char_crop
                # Make square 64x64
                char_square = make_square_with_padding(char_crop_inverted, target_size=64, pad_color=255)
                
                # Save crop
                crop_name = f"{img_basename}_plate0_char{idx}_{char_label}.png"
                crop_path = os.path.join(dataset_dir, char_label, crop_name)
                cv2.imwrite(crop_path, char_square)
                
                class_counts[char_label] += 1
                saved_total += 1
                
    print("\n--- Dataset Generation Summary ---")
    print(f"Total character crops saved: {saved_total}")
    print("Crops per class:")
    for char, count in sorted(class_counts.items()):
        print(f"Class '{char}': {count} images")

if __name__ == "__main__":
    main()
