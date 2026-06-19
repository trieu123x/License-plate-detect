from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2

detector = YOLO(r"C:\Users\admin\Downloads\code_ai\license-place-detector\detect\train\weights\best.pt")

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en"
)

img = cv2.imread(r"C:\Users\admin\Downloads\code_ai\license-place-detector\Dieu_0017.png")
img_display = img.copy()
results = detector(img,conf=0.5,iou=0.5)

for box in results[0].boxes.xyxy.cpu().numpy():

    print("crop")
    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]

    pad_x = int((x2 - x1) * 0.05)  
    pad_y = int((y2 - y1) * 0.05)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    plate = img[y1:y2, x1:x2]

    print("ocr start")

    result = ocr.predict(plate)
    
    print(result)

    print("ocr done")

    text = ""

    try:
        if result and len(result) > 0:
            # result[0] là dict chứa dữ liệu của ảnh đầu tiên
            data = result[0]
            
            # Lấy list chữ từ key 'rec_texts'
            if 'rec_texts' in data and data['rec_texts']:
                text = " ".join(data['rec_texts'])
    except Exception as e:
        print(f"Lỗi lấy text: {e}")
    
    print("Kết quả OCR:", text) 

    cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img_display,
        text,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


cv2.imshow("Result", img_display)

cv2.waitKey(0)
cv2.destroyAllWindows()