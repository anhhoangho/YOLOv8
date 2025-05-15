import cv2
import os
from ultralytics import YOLO


model = YOLO(r"E:\MY\PROJECT_PY_LTUD_CCN\LTUD_CONG\Yolov8_nhandienanh\Yolov8_nhandienanh\best.pt")


image_path = (r"E:\MY\PROJECT_PY_LTUD_CCN\LTUD_CONG\Yolov8_nhandienanh\Yolov8_nhandienanh\2.jpg")
image = cv2.imread(image_path)
if image is None:
    print(f"Không thể đọc hình ảnh tại: {image_path}")
    exit()


CONFIDENCE_THRESHOLD = 0.5
results = model(image_path, conf=0.3, iou=0.5, device="cpu")


class_names = ["Fork", "Spoon"]


count_fork = 0
count_spoon = 0
count_unknown = 0

detected_boxes = []


for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        aspect_ratio = (y2 - y1) / (x2 - x1)


        print(f"Class: {class_names[cls]}, Confidence: {conf:.2f}, Aspect Ratio: {aspect_ratio:.2f}")

        if cls < len(class_names) and conf >= CONFIDENCE_THRESHOLD:
            label = f"{class_names[cls]} ({conf:.2f})"
            color = (0, 255, 0)
            if class_names[cls] == "Fork" and conf >= 0.6:
                count_fork += 1
            elif class_names[cls] == "Spoon" and conf >= 0.5:
                count_spoon += 1
            else:
                label = f"UNKNOWN ({conf:.2f})"
                color = (0, 0, 255)
                count_unknown += 1
        else:
            label = f"UNKNOWN ({conf:.2f})"
            color = (0, 0, 255)
            count_unknown += 1


        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)


        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


        detected_boxes.append((x1, y1, x2, y2))


info_text = f"Fork: {count_fork} | Spoon: {count_spoon}"
info_text1 = f"UNKNOWN: {count_unknown}"


cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.putText(image, info_text1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imshow("Kết quả nhận diện", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

save_path = r"E:\MY\PROJECT_PY_LTUD_CCN\LTUD_CONG\Yolov8_nhandienanh\Yolov8_nhandienanh\test\anh34.jpg"
save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


cv2.imwrite(save_path, image)

print(f"Kết quả đã lưu tại: {save_path}")
print(f"Số lượng Fork: {count_fork}, Spoon: {count_spoon}, UNKNOWN: {count_unknown}")