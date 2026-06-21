# 🚗 Nhận diện & Đọc Biển Số Xe (License Plate Detection + OCR)

Hệ thống tự động phát hiện vùng biển số xe trong ảnh và đọc ký tự trên biển số, sử dụng pipeline kết hợp **YOLOv8** (phát hiện biển số) và **PaddleOCR** (nhận dạng ký tự).

---

## 📌 Mục lục

1. [Tổng quan pipeline](#1-tổng-quan-pipeline)
2. [Chi tiết từng bước](#2-chi-tiết-từng-bước)
3. [Dataset & Huấn luyện YOLO](#3-dataset--huấn-luyện-yolo)
4. [Kết quả và đánh giá](#4-kết-quả-và-đánh-giá)
5. [Cấu trúc thư mục](#5-cấu-trúc-thư-mục)
6. [Cách chạy](#6-cách-chạy)
7. [Deploy lên Hugging Face Spaces](#7-deploy-lên-hugging-face-spaces)

---

## 1. Tổng quan pipeline

Thuật toán đọc biển số xe được thực hiện qua **4 bước chính**:

```
Ảnh đầu vào
    │
    ▼
[Bước 1] Phát hiện vùng biển số  ← YOLOv8 (Object Detection)
    │
    ▼
[Bước 2] Cắt & căn chỉnh vùng biển số  ← Crop + Padding (OpenCV)
    │
    ▼
[Bước 3] Nhận dạng ký tự (OCR)  ← PaddleOCR (Text Recognition)
    │
    ▼
[Bước 4] Trực quan hoá & xuất kết quả  ← OpenCV + Gradio
```

---

## 2. Chi tiết từng bước

### 🔍 Bước 1 — Phát hiện vùng biển số (YOLOv8)

| Thông tin | Chi tiết |
|-----------|----------|
| **Kỹ thuật** | Object Detection với YOLOv8n (nano) |
| **Framework** | Ultralytics 8.4.71 |
| **Đầu vào** | Ảnh RGB gốc (bất kỳ kích thước) |
| **Đầu ra** | Tọa độ bounding box `(x1, y1, x2, y2)` của từng biển số |
| **Ngưỡng** | `conf = 0.5`, `iou = 0.5` (có thể tuỉnh chỉnh) |
| **Tốc độ** | ~1.8 ms/ảnh (inference, GPU Tesla T4) |

Mô hình nhận ảnh đầu vào, tự động resize về `896×896` (letterbox), sau đó dự đoán bounding box của tất cả biển số trong ảnh.

---

### ✂️ Bước 2 — Cắt & căn chỉnh vùng biển số (OpenCV)

| Thông tin | Chi tiết |
|-----------|----------|
| **Kỹ thuật** | Crop theo bounding box + thêm padding |
| **Thư viện** | OpenCV (`cv2`) |
| **Padding** | Mặc định `pad_ratio = 0.05` (5% chiều rộng/cao của box) — có thể chỉnh từ 0–30% |

**Mục đích:** Padding giúp tránh bị cắt mất các ký tự ở rìa biển số, cải thiện độ chính xác OCR. Vùng crop được giới hạn trong kích thước ảnh gốc để tránh tràn biên.

---

### 🔤 Bước 3 — Nhận dạng ký tự (PaddleOCR)

| Thông tin | Chi tiết |
|-----------|----------|
| **Kỹ thuật** | OCR đa giai đoạn (Text Detection + Recognition) |
| **Framework** | PaddleOCR (PP-OCRv4) |
| **Ngôn ngữ** | Tiếng Anh (`lang="en"`) — phù hợp biển số Latin |
| **Đầu vào** | Ảnh BGR đã crop từ bước 2 |
| **Đầu ra** | Chuỗi ký tự biển số (vd: `51F-123.45`) |
| **Cấu hình** | Tắt `doc_orientation_classify`, `doc_unwarping`, `textline_orientation` để tối ưu tốc độ |

PaddleOCR tự động thực hiện các bước nội bộ: phát hiện vùng text → nhận dạng từng dòng chữ → trả về chuỗi text.

---

### 🖼️ Bước 4 — Trực quan hoá & xuất kết quả (OpenCV + Gradio)

| Thông tin | Chi tiết |
|-----------|----------|
| **Kỹ thuật** | Vẽ bounding box + overlay text lên ảnh gốc |
| **Thư viện** | OpenCV (`cv2.rectangle`, `cv2.putText`) |
| **Giao diện** | Gradio Web UI (Hugging Face Spaces) |

Kết quả hiển thị: ảnh có bounding box màu xanh lá + text biển số số bên trên box; panel phụ liệt kê tất cả biển số đọc được.

---

## 3. Dataset & Huấn luyện YOLO

### 📂 Dataset: License Plate Detection Dataset (Ronak Gohil)

Dataset được lấy từ Kaggle: [`ronakgohil/license-plate-dataset`](https://www.kaggle.com/datasets/ronakgohil/license-plate-dataset), là các frame được trích xuất từ video camera giao thông.

| Tập dữ liệu | Số ảnh | Số nhãn (instances) | Ghi chú |
|-------------|--------|---------------------|---------|
| **Train** | 1,526 | 1,525 | 1 ảnh corrupt bị bỏ qua |
| **Validation (Val)** | 169 | 169 | Dùng để evaluate trong quá trình train |
| **Test** | Dùng tập Val | — | Không có tập test riêng; val set được dùng làm test |

- **Tổng cộng:** 1,695 ảnh
- **Số lớp:** 1 (`license_plate`)
- **Định dạng nhãn:** YOLO format (`.txt` với tọa độ normalized)
- **Nguồn ảnh:** Trích frame từ video 4, 5, 6, 8, 9 (camera giao thông thực tế)
- **Kích thước ảnh:** Đa dạng, từ ~300×600 đến ~1080×1920

---

### ⚙️ Cấu hình huấn luyện YOLO

```yaml
# license_plate.yaml
path: /kaggle/input/datasets/ronakgohil/license-plate-dataset/archive
train: images/train
val:   images/val
nc: 1
names:
  0: license_plate
```

| Hyperparameter | Giá trị | Mô tả |
|---------------|---------|-------|
| **Model** | `yolov8n.pt` | YOLOv8 Nano, pretrained trên COCO |
| **Epochs** | 100 | Tổng số epoch huấn luyện |
| **Image size** | 896 | Kích thước ảnh đầu vào |
| **Batch size** | 32 | Số ảnh mỗi batch |
| **Optimizer** | AdamW | Tối ưu hoá gradient |
| **Learning rate** | 5e-4 | Learning rate ban đầu |
| **LR scheduler** | Cosine (`cos_lr=True`) | Giảm dần theo cosine |
| **Weight decay** | 1e-3 | Regularization L2 |
| **Patience** | 30 | Dừng sớm nếu không cải thiện |
| **Device** | 2× Tesla T4 GPU | DDP (Distributed Data Parallel) |
| **Close mosaic** | epoch 85 | Tắt augment mosaic ở 15 epoch cuối |
| **AMP** | Bật (auto) | Automatic Mixed Precision |

---

### 🔀 Phương pháp tiền xử lý & Data Augmentation

#### Tiền xử lý cơ bản
- **Letterbox resize:** Resize ảnh về `896×896`, giữ tỷ lệ khung hình
- **Chuẩn hoá pixel:** `[0, 255] → [0, 1]`
- **BGR → RGB conversion** trong quá trình load

#### Data Augmentation (áp dụng tự động bởi Ultralytics)

**Augmentation hình học:**
| Kỹ thuật | Tham số | Mục đích |
|----------|---------|----------|
| `fliplr` | p=0.5 | Lật ngang ngẫu nhiên |
| `degrees` | ±5° | Xoay nhẹ |
| `translate` | 10% | Dịch chuyển vị trí |
| `scale` | 30% | Thay đổi kích thước |
| `mosaic` | p=1.0 (ep. 1–85) | Ghép 4 ảnh thành 1 |
| `mixup` | p=0.1 | Trộn 2 ảnh |

**Augmentation màu sắc:**
| Kỹ thuật | Tham số | Mục đích |
|----------|---------|----------|
| `hsv_h` | 0.015 | Biến thiên hue |
| `hsv_s` | 0.5 | Biến thiên saturation |
| `hsv_v` | 0.3 | Biến thiên brightness |

**Augmentation Albumentations (tự động thêm bởi Ultralytics):**
| Kỹ thuật | Xác suất | Mục đích |
|----------|----------|----------|
| `Blur` | p=0.01 | Giả lập ảnh mờ |
| `MedianBlur` | p=0.01 | Giả lập nhiễu camera |
| `ToGray` | p=0.01 | Giả lập ảnh đen trắng |
| `CLAHE` | p=0.01 | Tăng cường tương phản cục bộ |

---

## 4. Kết quả và đánh giá

### 📊 Metric đánh giá

Hệ thống sử dụng các metric chuẩn của COCO Object Detection:

| Metric | Ý nghĩa |
|--------|---------|
| **mAP50** | Mean Average Precision tại IoU ≥ 0.50 |
| **mAP50-95** | Mean Average Precision trung bình tại IoU từ 0.50 → 0.95 (bước 0.05) |
| **Precision (P)** | Tỷ lệ dự đoán đúng trong tổng số dự đoán dương |
| **Recall (R)** | Tỷ lệ phát hiện đúng trong tổng số ground truth |

### 🏆 Kết quả mô hình YOLO trên tập Validation

Kết quả tốt nhất (`best.pt`) sau 100 epoch:

| Metric | Giá trị |
|--------|---------|
| **mAP@50** | **99.46%** |
| **mAP@50–95** | **86.59%** |
| **Precision** | 99.3% |
| **Recall** | 99.4% |

> **Epoch đạt tốt nhất:** Epoch 73 với mAP50=99.5%, mAP50-95=86.5%

### 📈 Tiến trình huấn luyện (các mốc quan trọng)

| Epoch | mAP@50 | mAP@50-95 | Ghi chú |
|-------|--------|-----------|---------|
| 1 | 43.8% | 31.9% | Khởi đầu |
| 2 | 91.2% | 63.7% | Hội tụ nhanh |
| 4 | 98.6% | 71.4% | Ổn định |
| 15 | 99.3% | 80.3% | |
| 35 | 99.4% | 83.8% | |
| 56 | 99.4% | 84.7% | |
| **73** | **99.5%** | **86.5%** | 🏆 **Best checkpoint** |
| 100 | 99.3% | 82.8% | Epoch cuối |

- **Thời gian huấn luyện:** ~0.658 giờ (≈ 39 phút) trên 2× Tesla T4 GPU
- **Tốc độ inference:** 0.1ms preprocess + 1.8ms inference + 2.1ms postprocess per image

### 🔍 Kết quả phát hiện trên tập Validation (Predict)

Trên 169 ảnh validation:
- **168/169 ảnh** có biển số được phát hiện thành công
- **1 ảnh** không phát hiện được (`video8_1320.jpg` — trường hợp biển số bị khuất)
- Một số ảnh phát hiện được **2 biển số** cùng lúc (`video5_0`, `video6_220`, `video8_2460`)

### 📝 Đánh giá kết quả cuối cùng (End-to-End Pipeline)

| Thành phần | Kết quả |
|-----------|---------|
| **YOLO detect** (mAP@50) | **99.46%** |
| **YOLO detect** (mAP@50-95) | **86.59%** |
| **OCR (PaddleOCR)** | Phụ thuộc chất lượng ảnh crop; đạt tốt với biển số rõ ràng |
| **Pipeline tổng thể** | Hoạt động ổn định trên ảnh thực tế qua Gradio app |

> **Nhận xét:** Mô hình YOLO đạt độ chính xác rất cao (mAP@50 ≈ 99.5%) nhờ dataset đa dạng và chiến lược augmentation phong phú. Giai đoạn OCR phụ thuộc nhiều vào chất lượng biển số trong ảnh gốc (độ phân giải, góc chụp, ánh sáng).

---

## 5. Cấu trúc thư mục

```
license-place-detector/
├── app.py                        # Gradio app chính (pipeline đầy đủ)
├── best.pt                       # Weights YOLOv8 đã train (best checkpoint)
├── requirements.txt              # Thư viện Python cần cài
├── README.md                     # Tài liệu này
├── nhan-dien-bien-so-xe.ipynb    # Notebook Kaggle: train + validate YOLO
├── assets/                       # Ảnh mẫu để demo
│   ├── car.jpg
│   ├── car1.jpg
│   └── ...
├── License-plate/                # Thư mục con (phiên bản thứ hai của app)
│   ├── app.py
│   ├── best.pt
│   └── requirements.txt
└── detect/                       # Kết quả prediction (auto-generated)
```

---

## 6. Cách chạy

### Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### Chạy Gradio app

```bash
python app.py
```

Mở trình duyệt tại `http://localhost:7860`, upload ảnh xe và bấm **"Nhận diện"**.

### Tùy chỉnh nâng cao

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| Confidence threshold | 0.5 | Ngưỡng tin cậy YOLO (tăng để giảm false positive) |
| IoU threshold | 0.5 | Ngưỡng IoU cho NMS |
| Padding ratio | 0.05 | Padding quanh box trước khi OCR (0–30%) |

---

## 7. Deploy lên Hugging Face Spaces

1. Tạo Space mới, chọn **SDK: Gradio**.
2. Upload các file: `app.py`, `requirements.txt`, `README.md` và **`best.pt`** vào repo của Space.
3. Đợi Space build xong (cài `paddlepaddle` + `ultralytics` mất vài phút).
4. Mở app, upload ảnh xe và bấm **"Nhận diện"**.

---

## 📦 Requirements chính

```
ultralytics>=8.4.0     # YOLOv8
paddlepaddle>=2.5.0    # PaddlePaddle (backend cho PaddleOCR)
paddleocr>=2.7.0       # PaddleOCR
opencv-python>=4.6.0   # Xử lý ảnh
gradio>=4.0.0          # Web UI
```
