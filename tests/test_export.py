from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.export(format="engine", half = True)