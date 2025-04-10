import cv2
import json
import base64
import numpy as np
import os
from ultralytics import YOLO
from glob import glob

# Configuration
IMAGE_FOLDER = "/home/or22503/Louise_rat_tracking/auto_labelled_frames"
MODEL_PATH = "/home/or22503/Louise_rat_tracking/runs/detect/train4/weights/best.pt"
MIN_CONFIDENCE = 0.83

# Load YOLO model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = YOLO(MODEL_PATH)
labels = model.names

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Get all images in the folder
image_paths = glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + glob(os.path.join(IMAGE_FOLDER, "*.png"))

if not image_paths:
    raise FileNotFoundError(f"No images found in {IMAGE_FOLDER}")

for image_path in image_paths:
    print(f"Processing {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_path}, could not read image.")
        continue

    # Convert BGR to RGB to match YOLO model expectations
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape  # Get original image dimensions

    # Run YOLO inference with consistent image size
    results = model(image, verbose=False, imgsz=640)  # Use the same imgsz as in annotating_to_video.py

    # Reinitialize JSON data for each image
    json_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": encode_image(image_path),
        "imageHeight": height,
        "imageWidth": width,
    }

    for result in results:
        # YOLOv11 does not output masks, so we only process bounding boxes
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes

        for class_idx, score, box in zip(classes, scores, boxes):
            if score < MIN_CONFIDENCE:
                continue

            # Filter for the 'rat' class
            if labels[class_idx] != "rat":
                continue

            # Convert bounding box coordinates to Python float
            box = box.astype(float).tolist()

            # Add bounding box annotation with top-left and bottom-right coordinates
            shape_data_bbox = {
                "label": labels[class_idx],
                "points": [
                    [box[0], box[1]],  # Top-left corner
                    [box[2], box[3]]   # Bottom-right corner
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
            json_data["shapes"].append(shape_data_bbox)

    # Save JSON file with the same name as the image
    json_path = os.path.splitext(image_path)[0] + ".json"
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Saved annotation: {json_path}")

print("All images processed and labeled.")
