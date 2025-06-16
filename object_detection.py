import cv2
import time
from ultralytics import YOLO
import json

with open("yolo_classes.json") as fj:
    classNames = json.load(fj)

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

prevTime = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    results = model(frame,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),3)

            conf = int(box.conf[0]*100)
            obj = classNames["class"][str(int(box.cls[0]))]

            if conf >= 75:
                cv2.putText(frame,obj + " " + str(conf),(max(0,x1),max(40,y1-10)),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)



    #currTime = time.time()
    #fps = int(1/(currTime - prevTime))
    #prevTime = currTime
    #cv2.putText(frame,str(fps),(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    
    cv2.imshow("Video",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()