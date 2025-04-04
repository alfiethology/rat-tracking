import os
import cv2
import csv
import json
from tqdm import tqdm
from ultralytics import YOLO
import time
import datetime

# ====== Config ======
MODEL_PATH = "/home/or22503/Louise_rat_tracking/runs/detect/train2/weights/best.pt"
VIDEO_FOLDER = "/home/or22503/Louise_rat_tracking/videos/raw_videos"
AREAS_JSON_PATH = "/home/or22503/Louise_rat_tracking/areas.json"
SCHEDULE_JSON_PATH = "/home/or22503/Louise_rat_tracking/multi_video_time_stamps.json"
CSV_OUTPUT_FOLDER = "/home/or22503/Louise_rat_tracking/csv_outputs"
MIN_CONFIDENCE = 0.8
FRAME_SKIP = int(input("frame skip: "))  # Set this to 1 for every frame, 5 for every 5th frame, etc.

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

with open(AREAS_JSON_PATH, 'r') as f:
    area_boxes = json.load(f)

area_rects = {
    name: [make_rect(rect) for rect in rect_list]
    for name, rect_list in area_boxes.items()
}

def is_point_in_rect(point, rect):
    x, y = point
    return rect['xmin'] <= x <= rect['xmax'] and rect['ymin'] <= y <= rect['ymax']

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
        s['end_seconds'] = s['start_seconds'] + 300  # default to 5 minutes
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

            for name, rect_list in area_rects.items():
                for rect_obj in rect_list:  # rects are already precomputed
                    if is_point_in_rect((center_x, center_y), rect_obj):
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