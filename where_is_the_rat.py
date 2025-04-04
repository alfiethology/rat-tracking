import os
import cv2
import json
import csv
from ultralytics import YOLO

# === USER INPUT ===
MODEL_PATH = "/home/or22503/Louise_rat_tracking/runs/detect/train/weights/best.pt"
VIDEO_PATH = "/home/or22503/Louise_rat_tracking/videos/dumbledore.avi"
AREAS_PATH = "/home/or22503/Louise_rat_tracking/areas.json"  # Define 4 areas in here
RAT_NAME = "Dumbledore"  # Manually set this per video
OUTPUT_CSV = "rat_area_tracking.csv"
MIN_CONFIDENCE = 0.6

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)

# === LOAD AREAS CONFIG ===
with open(AREAS_PATH, 'r') as f:
    area_definitions = json.load(f)

# === OPEN VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {VIDEO_PATH}")

video_name = os.path.basename(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Processing video: {video_name} ({frame_count} frames)")

# === FUNCTION TO FIND AREA ===
def get_area_name(x_center, y_center):
    for area_name, (top_left, bottom_right) in area_definitions.items():
        x1, y1 = top_left
        x2, y2 = bottom_right
        if x1 <= x_center <= x2 and y1 <= y_center <= y2:
            return area_name
    return "middle"

# === PROCESS VIDEO ===
data_rows = []
frame_idx = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

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

            if best_score < MIN_CONFIDENCE:
                continue

            x1, y1, x2, y2 = boxes[best_idx]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            area_name = get_area_name(x_center, y_center)

            data_rows.append({
                "video_name": video_name,
                "rat_name": RAT_NAME,
                "frame_number": frame_idx,
                "area_occupied": area_name
            })

except KeyboardInterrupt:
    print("\n Interrupted by user. Saving collected data...")

finally:
    cap.release()
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "rat_name", "frame_number", "area_occupied"])
        writer.writeheader()
        writer.writerows(data_rows)
    print(f"Done! Results saved to {OUTPUT_CSV} ({len(data_rows)} frames processed)")

