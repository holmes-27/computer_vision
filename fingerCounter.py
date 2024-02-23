import cv2
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import mediapipe as mp
import time
from google.protobuf.json_format import MessageToDict

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,min_tracking_confidence=0.7,model_complexity=0)
draw = mp.solutions.drawing_utils

currTime = 0
prevTime = 0

# 1 -> open, 0 -> close
# finger order - thumb, index, middle, ring, pinky
finger = [1,1,1,1,1]
# [fingerId,x,y]
fingerList = []

# Hand classification
def getHand(res):
    if res.multi_handedness:
        for _,handClass in enumerate(res.multi_handedness):
            dic = MessageToDict(handClass)
            return dic["classification"][0]["label"].lower()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    height,width,_ = frame.shape

    result = hands.process(frameRGB)
    
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame,hand,mpHands.HAND_CONNECTIONS)
            for hid,lm in enumerate(hand.landmark):
                x = int(lm.x * width)
                y = int(lm.y * height)
                fingerList.append([hid,x,y])
                        
            if getHand(result) == "left":
                left = True
            else:
                left = False
                
            for i in range(4,21,4):
                if i == 4:
                    # Thumb finger (x-axis)
                    if left:
                        if fingerList[4-21][1] < fingerList[5-21][1]:
                            finger[i//4 - 1] = 0
                        else:
                            finger[i//4 - 1] = 1
                    else:
                        if fingerList[4-21][1] > fingerList[5-21][1]:
                            finger[i//4 - 1] = 0
                        else:
                            finger[i//4 - 1] = 1                                      
                else:
                    # Other fingers (y-axis)
                    if fingerList[i-21][2] > fingerList[(i-2)-21][2]:
                        finger[i//4 - 1] = 0
                    else:
                        finger[i//4 - 1] = 1
            
            cv2.rectangle(frame,(0,0),(120,120),(0,0,0),cv2.FILLED)            
            cv2.putText(frame, str(finger.count(1)), (30,85), cv2.FONT_HERSHEY_COMPLEX, 3, (255,255,255),2)   

    currTime = time.time()
    fps = int(1/(currTime - prevTime))
    prevTime = currTime
    cv2.putText(frame,str(fps),(width-100,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)

    cv2.imshow("Video",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()