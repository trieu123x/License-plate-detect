import os
import cv2
import numpy as np
import random

def make_square_with_padding(img, target_size=64, pad_color=255):
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
        
    h_sq, w_sq = squared_img.shape[:2]
    if h_sq < target_size:
        interp = cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_AREA
        
    return cv2.resize(squared_img, (target_size, target_size), interpolation=interp)

def generate_synthetic_char(char, target_size=64):
    img = np.ones((100, 100), dtype=np.uint8) * 255
    
    font = random.choice([
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_TRIPLEX
    ])
    
    font_scale = random.uniform(1.2, 1.5)
    thickness = random.randint(2, 3)
    
    if char == '4':
        # Draw open-top 4 to match Vietnamese plates
        w = random.randint(16, 20)
        h = random.randint(35, 42)
        cx = 50
        cy = 50
        
        # Horizontal crossbar
        x1 = cx - w // 2
        x2 = cx + w // 2
        y = cy + h // 6
        cv2.line(img, (x1, y), (x2, y), 0, thickness, lineType=cv2.LINE_AA)
        
        # Left vertical/diagonal stroke
        lx = cx - w // 3
        ly_top = cy - h // 2
        cv2.line(img, (lx, ly_top), (lx, y), 0, thickness, lineType=cv2.LINE_AA)
        
        # Right vertical stem
        rx = cx + w // 4
        ry_top = cy - h // 2
        ry_bottom = cy + h // 2
        cv2.line(img, (rx, ry_top), (rx, ry_bottom), 0, thickness, lineType=cv2.LINE_AA)
    else:
        (w, h), baseline = cv2.getTextSize(char, font, font_scale, thickness)
        x = (100 - w) // 2
        y = (100 + h) // 2 - baseline // 2
        cv2.putText(img, char, (x, y), font, font_scale, 0, thickness, lineType=cv2.LINE_AA)
        
    # Crop tightly to the character
    inverted = 255 - img
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cx, cy, cw, ch = cv2.boundingRect(contours[-1])
        crop = img[cy:cy+ch, cx:cx+cw]
    else:
        crop = img
        ch, cw = crop.shape[:2]
        
    # Squeeze horizontally to match the real plate aspect ratio of ~0.35
    target_aspect = 0.35
    new_h = ch
    new_w = int(ch * target_aspect)
    new_w = max(4, new_w)
    
    crop_narrow = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad character with 15% padding
    pad_h = int(new_h * 0.15)
    pad_w = int(new_w * 0.15)
    padded_char = cv2.copyMakeBorder(
        crop_narrow, pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_CONSTANT, value=255
    )
    
    # Make square with padding and resize to 64x64
    res = make_square_with_padding(padded_char, target_size=64, pad_color=255)
    
    # Apply random rotation
    angle = random.uniform(-10, 10)
    center = (target_size // 2, target_size // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    res = cv2.warpAffine(res, rot_mat, (target_size, target_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    # Apply slight blur
    blur_k = random.choice([0, 3])
    if blur_k > 0:
        res = cv2.GaussianBlur(res, (blur_k, blur_k), 0)
        
    return res

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")
    
    VALID_CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZ"
    
    print("--- Cleaning Old Synthetic Data & Generating New Narrow Synthetic Data ---")
    for char in VALID_CHARS:
        char_dir = os.path.join(dataset_dir, char)
        os.makedirs(char_dir, exist_ok=True)
        
        # 1. Clean old synthetic images
        for file in os.listdir(char_dir):
            if "synthetic_" in file:
                os.remove(os.path.join(char_dir, file))
                
        # 2. Count remaining real images
        current_count = len(os.listdir(char_dir))
        if current_count < 100:
            to_generate = 100 - current_count
            print(f"Class '{char}': real count = {current_count}. Generating {to_generate} narrow synthetic images...")
            
            for i in range(to_generate):
                synth_img = generate_synthetic_char(char)
                img_name = f"synthetic_{i}.png"
                cv2.imwrite(os.path.join(char_dir, img_name), synth_img)
                
    print("Synthetic data regeneration completed successfully! All classes have 100 images.")

if __name__ == "__main__":
    main()
