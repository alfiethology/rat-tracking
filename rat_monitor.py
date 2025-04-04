import os
import cv2
import csv
import json
from tqdm import tqdm
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

# ====== Load YOLO Model ======
model = YOLO(MODEL_PATH)

# ====== Load Area Boxes ======
def make_rect(coords):
    (x1, y1), (x2, y2) = coords
    return {"xmin": min(x1, x2), "xmax": max(x1, x2), "ymin": min(y1, y2), "ymax": max(y1, y2)}

def is_point_in_rect(point, rect):
    x, y = point
    return rect['xmin'] <= x <= rect['xmax'] and rect['ymin'] <= y <= rect['ymax']

def is_point_in_polygon(point, polygon_coords):
    polygon = Polygon(polygon_coords)
    return polygon.contains(Point(point))

with open(AREAS_JSON_PATH, 'r') as f:  # Update to use areas_new.json
    area_boxes = json.load(f)

area_shapes = {}
for name, shape_list in area_boxes.items():
    area_shapes[name] = []
    for shape_coords in shape_list:
        if len(shape_coords) == 2:  # Rectangle (two points)
            area_shapes[name].append({"type": "rectangle", "shape": make_rect(shape_coords)})
        elif len(shape_coords) > 2:  # Polygon (multiple points)
            area_shapes[name].append({"type": "polygon", "shape": Polygon(shape_coords)})
        else:
            raise ValueError(f"Invalid shape definition for area '{name}'")

# ====== Load Rat Schedules ======
def hms_to_seconds(hms_str):
    h, m, s = map(int, hms_str.split(":"))
    return h * 3600 + m * 60 + s

with open(SCHEDULE_JSON_PATH, 'r') as f:
    full_schedule = json.load(f)

schedule_dict = {}
for entry in full_schedule:
    video = entry['video_name']
    sched = entry['schedule']
    for s in sched:
        s['start_seconds'] = hms_to_seconds(s['start_time'])
        s['end_seconds'] = s['start_seconds'] + 300  # 5 minutes
    schedule_dict[video] = sched

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

    # Prepare CSV output file
    video_basename = os.path.splitext(video_file)[0]
    csv_output_path = os.path.join(CSV_OUTPUT_FOLDER, f"{video_basename}_{date}.csv")
    csv_file = open(csv_output_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "timestamp", "rat_name", "area"])

    # Progress bar
    for frame_idx in tqdm(range(1, frame_count + 1, FRAME_SKIP), desc="🔍 Detecting", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = frame_idx / fps
        rat_name = get_current_rat_name(rat_schedule, current_time_sec)
        if rat_name is None:
            continue  # Skip frame

        # Timestamp
        hrs = int(current_time_sec // 3600)
        mins = int((current_time_sec % 3600) // 60)
        secs = int(current_time_sec % 60)
        timestamp = f"{hrs:02}:{mins:02}:{secs:02}"

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
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            for name, shapes in area_shapes.items():
                for shape_data in shapes:
                    if shape_data["type"] == "rectangle":
                        if is_point_in_rect((center_x, center_y), shape_data["shape"]):
                            area_label = name
                            break
                    elif shape_data["type"] == "polygon":
                        if is_point_in_polygon((center_x, center_y), shape_data["shape"]):
                            area_label = name
                            break
                if area_label != "none":
                    break
        else:
            continue  # No good detection

        # Write to CSV
        csv_writer.writerow([frame_idx, timestamp, rat_name, area_label])

    cap.release()
    csv_file.close()
    print(f"CSV saved to: {csv_output_path}")

print("All videos processed.")

end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)
print(f"\n All videos processed in {int(mins)} minutes and {int(secs)} seconds.")