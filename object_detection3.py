# object detection with central crosshair

import cv2
import time
import json
from ultralytics import YOLO

# Load class names
with open("yolo_classes.json") as fj:
    classNames = json.load(fj)

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

prev_time = 0

def draw_crosshair(frame, color=(0, 255, 0), size=20, thickness=2):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    # Horizontal
    cv2.line(frame, (center[0] - size, center[1]), (center[0] + size, center[1]), color, thickness)
    # Vertical
    cv2.line(frame, (center[0], center[1] - size), (center[0], center[1] + size), color, thickness)
    return center

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    center_point = draw_crosshair(frame)

    detected = False

    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = int(box.conf[0] * 100)
            cls_id = int(box.cls[0])
            class_name = classNames["class"].get(str(cls_id), "Unknown")

            if conf >= 75:
                # Check if center point is inside the bounding box
                if x1 <= center_point[0] <= x2 and y1 <= center_point[1] <= y2:
                    detected = True
                    # Draw cinematic-style target lock
                    color = (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{class_name} {conf}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.9,
                        color,
                        2
                    )

    # Show "locked" or "searching" near center
    status_text = "LOCKED" if detected else "SEARCHING..."
    cv2.putText(
        frame,
        status_text,
        (center_point[0] - 80, center_point[1] + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255) if detected else (150, 150, 150),
        2
    )

    # FPS display
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)

    cv2.imshow("Target Lock - AI Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
