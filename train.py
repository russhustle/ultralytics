from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load model
results = model.train(
    data="ultralytics/cfg/datasets/coco8.yaml",
    epochs=3,
    optimizer="Lion",  # New optimizer
)
