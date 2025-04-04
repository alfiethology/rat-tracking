import os
import cv2
import random

number_of_frames = 500

def get_all_frames_info(input_folder):
    all_frames = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp4") or file_name.endswith(".avi"):
            video_path = os.path.join(input_folder, file_name)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for idx in range(total_frames):
                all_frames.append((video_path, idx))
    return all_frames

def extract_random_frames(all_frames, output_folder, num_frames=number_of_frames):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    selected = random.sample(all_frames, min(num_frames, len(all_frames)))

    for i, (video_path, frame_idx) in enumerate(selected):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_name = os.path.join(output_folder, f"{base_name}_frame_{frame_idx}.jpg")
            cv2.imwrite(frame_name, frame)
        cap.release()

# Usage
input_folder = "videos"
output_folder = "/home/or22503/Louise_rat_tracking/frames"

all_frames = get_all_frames_info(input_folder)
extract_random_frames(all_frames, output_folder, num_frames=number_of_frames)
