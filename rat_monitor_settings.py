MODEL_PATH = "/home/or22503/Louise_rat_tracking/runs/detect/train2/weights/best.pt"
VIDEO_FOLDER = "/home/or22503/Louise_rat_tracking/videos/raw_videos"
AREAS_JSON_PATH = "/home/or22503/Louise_rat_tracking/areas.json"
SCHEDULE_JSON_PATH = "/home/or22503/Louise_rat_tracking/Batch_1_2024_EPM.json"
CSV_OUTPUT_FOLDER = "/home/or22503/Louise_rat_tracking/csv_outputs"
MIN_CONFIDENCE = 0.8
FRAME_SKIP = 100  # Set this to 1 for every frame, 5 for every 5th frame, etc.
PERCENTAGE = 80  # Percentage of area occupied to be considered "in" the area