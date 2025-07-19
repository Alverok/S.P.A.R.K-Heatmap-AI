import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

video_path = 'C:/Users/Akhil/Desktop/Akhil/Amrita University/Programs/spark-valhalla/data/samples/crowd.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

heatmap = np.zeros((height, width), dtype=np.float32)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('C:/Users/Akhil/Desktop/Akhil/Amrita University/Programs/spark-valhalla/data/samples/crowd_heat.mp4', fourcc, fps, (width, height))

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
    print(f'Processing frame {frame_num}/{frame_count}...')

    results = model(frame)
    boxes = results[0].boxes
    person_mask = boxes.cls == 0
    person_boxes = boxes.xyxy[person_mask]

    for box in person_boxes:
        x1, y1, x2, y2 = box.int().tolist()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if 0 <= cx < width and 0 <= cy < height:
            # Add a very small increment to heatmap
            cv2.circle(heatmap, (cx, cy), radius=10, color=0.2, thickness=-1)

    heatmap_blurred = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=30, sigmaY=30)
    heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

    # Blend with original frame with very low heatmap opacity
    overlay = cv2.addWeighted(frame, 0.9, heatmap_color, 0.1, 0)
    out.write(overlay)

cap.release()
out.release()
