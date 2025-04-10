import os
import cv2
import csv
import json
from tqdm import tqdm  # Re-added tqdm
from ultralytics import YOLO
import time
import datetime
from rat_monitor_settings import *
from shapely.geometry import Point, Polygon  # Add this for polygon handling

# ====== Start Timer & date ======
date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
start_time = time.time()

# ====== Ensure output folder exists ======
os.makedirs(CSV_OUTPUT_FOLDER, exist_ok=True)

# Prepare single CSV output file
combined_csv_path = os.path.join(CSV_OUTPUT_FOLDER, f"combined_rat_data_{date}.csv")
combined_csv_file = open(combined_csv_path, mode='w', newline='')
combined_csv_writer = csv.writer(combined_csv_file)
combined_csv_writer.writerow(["EPM_session", "video_name", "rat_name", "timestamp", "frame_number", "area_occupied"])  # Added timestamp

# ====== Load YOLO Model ======
model = YOLO(MODEL_PATH)

# ====== Load Area Boxes ======
with open(AREAS_JSON_PATH, 'r') as f:
    area_boxes = json.load(f)

area_shapes = {}
for name, shape_list in area_boxes.items():
    area_shapes[name] = [Polygon(shape_coords) for shape_coords in shape_list]  # Directly load polygons

# Check if a point is inside any polygon
def is_point_in_any_polygon(point, polygons):
    point_obj = Point(point)
    return any(polygon.contains(point_obj) for polygon in polygons)

# Function to calculate the percentage of a bounding box that overlaps with a polygon
def calculate_overlap_percentage(bbox, polygon):
    bbox_polygon = Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])])
    intersection_area = bbox_polygon.intersection(polygon).area
    bbox_area = bbox_polygon.area
    return (intersection_area / bbox_area) * 100 if bbox_area > 0 else 0

# Function to determine the area occupied by the rat based on percentage overlap
def determine_area(bbox, area_shapes, current_area):
    max_overlap = 0
    new_area = "none"  # Default to 'none'

    for area_name, polygons in area_shapes.items():
        total_overlap = sum(calculate_overlap_percentage(bbox, polygon) for polygon in polygons)
        print(f"Debug: Area '{area_name}' overlap = {total_overlap:.2f}%")  # Debugging overlap percentages
        if total_overlap > max_overlap:
            max_overlap = total_overlap
            new_area = area_name

    # Check if the rat has moved to a new area
    if max_overlap >= PERCENTAGE:
        return new_area
    elif max_overlap > 0:  # Fallback to the area with the highest overlap
        return new_area
    return "none"

# ====== Load Rat Schedules ======
def hms_to_seconds(hms_str):
    h, m, s = map(int, hms_str.split(":"))
    return h * 3600 + m * 60 + s

with open(SCHEDULE_JSON_PATH, 'r') as f:
    full_schedule = json.load(f)

schedule_dict = {}
for entry in full_schedule:
    session = entry['EPM_session']
    video = entry['video_name']
    sched = entry['schedule']
    for s in sched:
        s['start_seconds'] = hms_to_seconds(s['start_time'])
        s['end_seconds'] = s.get('end_time') and hms_to_seconds(s['end_time']) or (s['start_seconds'] + 300)  # Default to 5 minutes
    schedule_dict[video] = {"session": session, "schedule": sched}

def get_current_rat_name(schedule, seconds_elapsed):
    for entry in schedule:
        if entry['start_seconds'] <= seconds_elapsed <= entry['end_seconds']:
            return entry['rat_name']
    return None

# ====== Process Each Video ======
video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]

for video_file in video_files:
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    print(f"\nProcessing {video_file}")

    if video_file not in schedule_dict:
        print(f"Skipping {video_file}: no schedule found")
        continue

    rat_schedule = schedule_dict[video_file]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_file}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Found {frame_count} frames at {fps:.2f} FPS")

    # Process frames with progress bar
    current_area = "none"  # Track the current area the rat is in

    for frame_idx in tqdm(range(1, frame_count + 1, FRAME_SKIP), desc="🔍 Detecting", unit="frame"):  # Re-added tqdm
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = frame_idx / fps
        rat_name = get_current_rat_name(schedule_dict[video_file]["schedule"], current_time_sec)
        if rat_name is None:
            continue  # Skip frame

        # Timestamp
        hrs = int(current_time_sec // 3600)
        mins = int((current_time_sec % 3600) // 60)
        secs = int(current_time_sec % 60)
        timestamp = f"{hrs:02}:{mins:02}:{secs:02}"  # Generate timestamp

        results = model(frame, verbose=False, imgsz=640)

        area_label = "none"
        best_box = None

        # Get best detection
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            if len(scores) == 0:
                continue
            best_idx = scores.argmax()
            if scores[best_idx] >= MIN_CONFIDENCE:
                best_box = boxes[best_idx]
                break

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            bbox = (x1, y1, x2, y2)  # Bounding box coordinates

            # Determine the area based on percentage overlap
            area_label = determine_area(bbox, area_shapes, current_area)
            current_area = area_label  # Update the current area
        else:
            continue  # No good detection

        # Write to combined CSV
        epm_session = schedule_dict[video_file]["session"]
        combined_csv_writer.writerow([epm_session, video_file, rat_name, timestamp, frame_idx, area_label])  # Ensure all areas, including 'middle', are written

    cap.release()
    print(f"Finished processing {video_file}")

combined_csv_file.close()
print(f"Combined CSV saved to: {combined_csv_path}")

end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)
print(f"\n All videos processed in {int(mins)} minutes and {int(secs)} seconds.")