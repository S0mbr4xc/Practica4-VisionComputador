# export_raw.py
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, simplify=False)  # ojo: simplify=False
print("✅ best.onnx exportado (raw)")
