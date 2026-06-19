# License Plate Detection + OCR

Gradio app sử dụng YOLO (Ultralytics) để phát hiện vùng biển số xe và PaddleOCR để đọc text trên biển số.

## Cấu trúc thư mục

```
.
├── app.py             # Gradio app chính
├── best.pt             # Weights YOLO đã train (BẮT BUỘC, đặt cùng cấp với app.py)
├── requirements.txt    # Thư viện cần cài
└── README.md
```

## Cách deploy lên Hugging Face Spaces

1. Tạo Space mới, chọn **SDK: Gradio**.
2. Upload các file: `app.py`, `requirements.txt`, `README.md` và **file weights `best.pt`** vào repo của Space (kéo thả trên web UI hoặc qua `git`/`huggingface_hub`).
3. Đợi Space build xong (cài `paddlepaddle` + `ultralytics` mất vài phút).
4. Mở app, upload ảnh xe và bấm "Nhận diện".
