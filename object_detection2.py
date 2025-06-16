# object detection with better UI

import cv2
import time
import random
import json
from ultralytics import YOLO

# Load class names
with open("yolo_classes.json") as fj:
    classNames = json.load(fj)

# Generate random colors for each class (optional: consistent palette)
def generate_class_colors(classes):
    return {cls: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cls in classes}

colors = generate_class_colors(classNames["class"])

# Load model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

prev_time = 0

def draw_label(img, text, pos, bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]

    x, y = pos
    cv2.rectangle(img, (x, y - text_size[1] - 10), (x + text_size[0] + 10, y), bg_color, -1)
    cv2.putText(img, text, (x + 5, y - 5), font, scale, (255, 255, 255), thickness)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = int(box.conf[0] * 100)
            cls_id = int(box.cls[0])
            class_name = classNames["class"].get(str(cls_id), "Unknown")
            color = colors.get(str(cls_id), (0, 255, 0))

            if conf >= 75:
                # Draw rounded rectangle (basic)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Label with class name and confidence
                label = f"{class_name} {conf}%"
                draw_label(frame, label, (x1, y1), color)

    # FPS display
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 2)

    # Show the frame
    cv2.imshow("Enhanced Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
