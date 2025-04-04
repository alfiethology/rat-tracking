import json
import os

# Paths to the JSON files and where to store the output labels
json_dir = '/home/or22503/Louise_rat_tracking/temp'  # Directory with JSONs
output_dir = '/home/or22503/Louise_rat_tracking/temp'  # Output YOLO .txt folder

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Class mapping
class_map = {"rat": 0}  # Add more classes as needed

# Process each JSON file
for json_file in os.listdir(json_dir):
    if json_file.endswith(".json"):
        json_path = os.path.join(json_dir, json_file)

        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Prepare output .txt path
        txt_filename = os.path.join(output_dir, json_file.replace('.json', '.txt'))

        # Open YOLO .txt file for writing
        with open(txt_filename, 'w') as f_out:
            shapes = data.get('shapes', [])
            if not shapes:
                print(f"No shapes in {json_file}")
                continue

            # You may also extract width/height from the JSON if it's included
            img_width = data['imageWidth']  # Replace with actual width if needed
            img_height = data['imageHeight']  # Replace with actual height if needed

            for shape in shapes:
                class_name = shape['label']
                points = shape['points']

                if not points or len(points) < 2:
                    print(f"Skipping invalid shape in {json_file}")
                    continue

                class_id = class_map.get(class_name, -1)
                if class_id == -1:
                    print(f"Unknown class '{class_name}' in {json_file}")
                    continue

                # Get bounding box coordinates
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)

                # Convert to YOLO format
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                bbox_width = (xmax - xmin) / img_width
                bbox_height = (ymax - ymin) / img_height

                # Write to file
                f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

print("Conversion to YOLO bounding box format completed!")
