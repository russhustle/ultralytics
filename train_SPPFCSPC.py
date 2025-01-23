from ultralytics import YOLO

model = YOLO(model="yolov8x-pose-SPPFCSPC.yaml")

results = model.train(data="coco8-pose.yaml", epochs=1, imgsz=640, batch=1)
