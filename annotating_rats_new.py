import os
import cv2
import json
import numpy as np
from ultralytics import YOLO

# Paths
MODEL_PATH = "/home/or22503/Louise_rat_tracking/runs/detect/train2/weights/best.pt"
INPUT_VIDEO_PATH = "/home/or22503/Louise_rat_tracking/videos/dumbledore.avi"
OUTPUT_VIDEO_PATH = "/home/or22503/Louise_rat_tracking/videos/annotated_Dumbledore.mp4"
AREAS_JSON_PATH = "/home/or22503/Louise_rat_tracking/areas.json"
RAT_NAME = "Dumbledore"

# Confidence threshold
MIN_CONFIDENCE = 0.6

# Load YOLO model
model = YOLO(MODEL_PATH)

# Load area definitions from JSON
with open(AREAS_JSON_PATH, 'r') as f:
    area_boxes = json.load(f)

# Convert area boxes to simple min/max format
def make_rect(area_coords):
    (x1, y1), (x2, y2) = area_coords
    return {
        "xmin": min(x1, x2),
        "xmax": max(x1, x2),
        "ymin": min(y1, y2),
        "ymax": max(y1, y2)
    }

area_rects = {name: make_rect(coords) for name, coords in area_boxes.items()}

# Helper: Check if a point is inside a bounding rectangle
def is_point_in_rect(point, rect):
    x, y = point
    return rect['xmin'] <= x <= rect['xmax'] and rect['ymin'] <= y <= rect['ymax']

# Open video
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {INPUT_VIDEO_PATH}")

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {frame_count} frames...")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    print(f"Processing frame {frame_idx}/{frame_count}", end='\r')

    results = model(frame, verbose=False)

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        if len(scores) == 0:
            continue

        best_idx = scores.argmax()
        best_score = scores[best_idx]

        if best_score >= MIN_CONFIDENCE:
            x1, y1, x2, y2 = map(int, boxes[best_idx])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            location_label = "unknown"
            for name, rect in area_rects.items():
                if is_point_in_rect((center_x, center_y), rect):
                    location_label = name
                    break

            label = f"{RAT_NAME} {location_label} ({best_score:.2f})"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate label size
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_x = x1
            label_y = max(y1 - 10, label_height + 10)

            # Draw background rectangle for text
            cv2.rectangle(frame,
                        (label_x, label_y - label_height - baseline),
                        (label_x + label_width, label_y + baseline),
                        (0, 255, 0),  # same color as box
                        thickness=-1)  # filled rectangle

            # Draw the text on top
            cv2.putText(frame, label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2, cv2.LINE_AA)  # black text with anti-aliasing


    out.write(frame)

cap.release()
out.release()
print(f"Annotated video saved to: {OUTPUT_VIDEO_PATH}")
