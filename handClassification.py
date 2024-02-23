import cv2
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import mediapipe as mp
import time
from google.protobuf.json_format import MessageToDict

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2,model_complexity=0,min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

currTime = 0
prevTime = 0

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = hands.process(frameRGB)
    
    if result.multi_handedness:
        if len(result.multi_handedness) == 2:
            cv2.putText(frame,"Both Hands", (10,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255),2)
        else:
            for _, handClass in enumerate(result.multi_handedness):
                dic = MessageToDict(handClass)
                txt = dic["classification"][0]["label"]
                cv2.putText(frame,txt + " Hand", (10,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255),2)
    
    currTime = time.time()
    fps = int(1/(currTime - prevTime))
    prevTime = currTime
    cv2.putText(frame,str(fps), (500,70), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255),2)
    
    cv2.imshow("Video",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()