import os
import zipfile
from ultralytics import YOLO


zip_path = r"E:\MY\PROJECT_PY_LTUD_CCN\LTUD_CONG\Yolov8_nhandienanh\Yolov8_nhandienanh\Fork_Spoon.zip"
extract_path = r"E:\MY\PROJECT_PY_LTUD_CCN\LTUD_CONG\Yolov8_nhandienanh\Yolov8_nhandienanh\data"

print("Giải nén dữ liệu...")
os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)
print("Giải nén hoàn tất!")


yaml_path = r"D:/pycharm/Yolov8_nhandienanh/data/data.yaml"


fixed_extract_path = extract_path.replace("\\", "/")


yaml_content = f"""train: {fixed_extract_path}/train/images
val: {fixed_extract_path}/valid/images
test: {fixed_extract_path}/test/images

nc: 2
names:
  0: Fork
  1: Spoon
"""

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"Đã tạo file YAML tại: {yaml_path}")

model = YOLO("yolov8n.pt")
model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,
    workers=4
)
print("Huấn luyện hoàn tất!")
