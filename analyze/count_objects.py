import torch
import cv2
import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
video_path = "/my/path/to/video.mp4"
output_csv = "object_counts.csv"  # Output
model_name = "yolov5l"  # yolov5s, yolov5m, yolov5l, yolov5x
target_classes = ['person', 'car', 'others']
confidence_threshold = 0.25
# -----------------------------

model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
model.conf = confidence_threshold

cap = cv2.VideoCapture(video_path)
frame_index = 0
results_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    detections = results.pandas().xyxy[0]
    counts = {cls: 0 for cls in target_classes}
    for _, row in detections.iterrows():
        if row['name'] in counts and row['name'] != 'others':
            counts[row['name']] += 1
    # Count 'others' as all objects not in other target classes
    counts['others'] = sum(1 for _, row in detections.iterrows() if row['name'] not in target_classes or row['name'] == 'others')
    row_data = [frame_index] + [counts[cls] for cls in target_classes]
    results_data.append(row_data)
    frame_index += 1
cap.release()

columns = ['Frame_Index'] + target_classes
df = pd.DataFrame(results_data, columns=columns)
df.to_csv(output_csv, index=False)

print(f"Done! Results saved to {output_csv}")
