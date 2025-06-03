import cv2
import os
import numpy as np
from ultralytics import YOLO

# Paths
video_path = "computer_vision/IMG_5402.MOV"
weapon_model_path = "computer_vision/best.pt"
pose_model_path = "computer_vision/yolo11m-pose.pt"
output_path = "video_output/annotated_video_2.mp4"

# Load models
weapon_model = YOLO(weapon_model_path)
pose_model = YOLO(pose_model_path)

# IOU Function
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    weapon_results = weapon_model.predict(frame, conf=0.5, imgsz=640)[0]
    pose_results = pose_model.predict(frame, conf=0.5, imgsz=640)[0]

    # Pose visualization (skeletons + person boxes)
    img_visual = pose_results.plot()

    # Draw weapon boxes
    for box, cls, conf in zip(weapon_results.boxes.xyxy.cpu().numpy(),
                              weapon_results.boxes.cls.cpu().numpy(),
                              weapon_results.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        label = f"{weapon_model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img_visual, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_visual, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw POIs
    weapon_boxes = weapon_results.boxes.xyxy.cpu().numpy()
    person_boxes = pose_results.boxes.xyxy.cpu().numpy()

    for px1, py1, px2, py2 in person_boxes:
        person_box = [px1, py1, px2, py2]
        is_poi = any(compute_iou(person_box, weapon_box) > 0.2 for weapon_box in weapon_boxes)
        if is_poi:
            cv2.rectangle(img_visual, (int(px1), int(py1)), (int(px2), int(py2)), (0, 125, 255), 2)
            cv2.putText(img_visual, "POI", (int(px1), int(py1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out.write(img_visual)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Video saved to: {output_path}")
