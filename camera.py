import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(
    r"E:\MY\PROJECT_PY_LTUD_CCN\LTUD_CONG\Yolov8_nhandienanh\Yolov8_nhandienanh\best.pt"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được khung hình từ camera")
        break


    results = model(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    class_names = ["FORK", "SPOON"]
    detected_boxes = []


    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls < len(class_names):
                label = f"{class_names[cls]} ({conf:.2f})"
            else:
                label = "UNKNOWN"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            detected_boxes.append((x1, y1, x2, y2))

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 5000:
            is_detected = any(
                (x < x2 and x + w > x1 and y < y2 and y + h > y1) for x1, y1, x2, y2 in detected_boxes
            )
            if not is_detected:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "UNKNOWN", (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)

    cv2.imshow("Detect từ camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
