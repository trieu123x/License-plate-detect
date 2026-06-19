"""
License Plate Detector + OCR — Gradio app cho Hugging Face Spaces

Pipeline:
1. YOLO (ultralytics) detect vùng biển số trong ảnh
2. Crop vùng biển số (có padding)
3. PaddleOCR đọc text trên vùng crop
4. Vẽ bounding box + text lên ảnh gốc, trả về cho user

Yêu cầu file:
- best.pt          : weights YOLO, đặt cùng thư mục với app.py
- requirements.txt : các package cần cài
"""

import os
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ----------------------------------------------------------------------------
# Load model 1 lần duy nhất khi app khởi động (KHÔNG load lại trong hàm xử lý,
# nếu không mỗi lần user bấm Submit sẽ rất chậm)
# ----------------------------------------------------------------------------

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

print("Loading YOLO model...")
detector = YOLO(WEIGHTS_PATH)

print("Loading PaddleOCR model...")
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
)

print("Models loaded. App is ready.")


# ----------------------------------------------------------------------------
# Hàm xử lý chính
# ----------------------------------------------------------------------------

def detect_and_read_plate(image, conf, iou, pad_ratio):
    """
    image: ảnh dạng numpy array (RGB) do Gradio truyền vào
    conf, iou: ngưỡng cho YOLO
    pad_ratio: tỉ lệ padding quanh box trước khi crop để OCR

    Trả về:
    - ảnh đã vẽ box + text (numpy array, RGB)
    - chuỗi text tổng hợp tất cả biển số đọc được (hiển thị trong textbox)
    """
    if image is None:
        return None, "Chưa có ảnh đầu vào."

    # Gradio trả ảnh dạng RGB, OpenCV làm việc với BGR -> convert
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_display = img_bgr.copy()

    results = detector(img_bgr, conf=conf, iou=iou, verbose=False)

    h, w = img_bgr.shape[:2]
    plates_text = []

    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        # Không phát hiện được biển số nào
        img_rgb_out = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        return img_rgb_out, "Không phát hiện được biển số nào trong ảnh."

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        pad_x = int((x2 - x1) * pad_ratio)
        pad_y = int((y2 - y1) * pad_ratio)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        plate = img_bgr[y1:y2, x1:x2]

        text = ""
        if plate.size > 0:
            try:
                ocr_result = ocr.predict(plate)
                if ocr_result and len(ocr_result) > 0:
                    data = ocr_result[0]
                    if "rec_texts" in data and data["rec_texts"]:
                        text = " ".join(data["rec_texts"])
            except Exception as e:
                text = f"[Lỗi OCR: {e}]"

        plates_text.append(text if text else "(không đọc được)")

        # Vẽ box
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Vẽ text phía trên box (nếu box quá sát mép trên thì vẽ xuống dưới)
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 25
        cv2.putText(
            img_display,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    img_rgb_out = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    summary = "\n".join(f"Biển số {i+1}: {t}" for i, t in enumerate(plates_text))

    return img_rgb_out, summary


# ----------------------------------------------------------------------------
# Giao diện Gradio
# ----------------------------------------------------------------------------

with gr.Blocks(title="License Plate Detection + OCR") as demo:
    gr.Markdown(
        """
        # 🚗 Nhận diện biển số xe (YOLO + PaddleOCR)
        Upload ảnh xe → model sẽ detect vùng biển số và đọc text trên biển số.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Ảnh đầu vào", type="numpy")

            with gr.Accordion("Tùy chỉnh nâng cao", open=False):
                conf_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                    label="Confidence threshold (YOLO)"
                )
                iou_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                    label="IoU threshold (YOLO)"
                )
                pad_slider = gr.Slider(
                    minimum=0.0, maximum=0.3, value=0.05, step=0.01,
                    label="Padding quanh box trước khi OCR"
                )

            submit_btn = gr.Button("Nhận diện", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Kết quả")
            output_text = gr.Textbox(label="Text biển số đọc được", lines=5)

    submit_btn.click(
        fn=detect_and_read_plate,
        inputs=[input_image, conf_slider, iou_slider, pad_slider],
        outputs=[output_image, output_text],
    )


    example_dir = os.path.join(os.path.dirname(__file__), "assets")
    example_files = [
        os.path.join(example_dir, fname)
        for fname in ["car.jpg", "car1.jpg", "car2.png", "car3.png", "car4.jpg"]
        if os.path.exists(os.path.join(example_dir, fname))
    ]

    if example_files:
        gr.Examples(
            examples=example_files,
            inputs=input_image,
            label="Ảnh mẫu (bấm để test nhanh)",
        )

if __name__ == "__main__":
    demo.launch()