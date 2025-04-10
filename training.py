from ultralytics import YOLO

# Load a YOLOv8 large model pre-trained on COCO dataset
model = YOLO('/home/or22503/Louise_rat_tracking/models/yolo11x.pt')  # Change from yolov8n.pt to yolov8l.pt for the large model

# Train the model on your dataset
model.train(
    data='/home/or22503/Louise_rat_tracking/define_structure.yaml',  # Path to your dataset YAML
    epochs=1000,  # Number of epochs
    imgsz=(640),  # Target image size; adjust according to your needs
    batch=16,  # Batch size; adjust based on your hardware
    patience=200
)