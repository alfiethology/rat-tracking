from ultralytics import YOLO

# Load your trained model
model = YOLO("/home/or22503/Louise_rat_tracking/runs/detect/train2/weights/best.pt")

# Run validation (this assumes your model was trained using the YOLO format)
metrics = model.val()  # You can pass data=... if needed

# Print evaluation results
print(metrics)
