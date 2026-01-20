from ultralytics import YOLO

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")


model.train(
    data="Dataset/data.yaml",   
    epochs=50,
    imgsz=640,
    batch=16,
    name="indian_number_plate"
)
