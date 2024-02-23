import cv2
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,model_complexity=0,min_detection_confidence=0.7)

x1 = 0
x2 = 0
cou = 1

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape
    
    result = hands.process(frameRGB)
    
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            for hid,lm in enumerate(hand.landmark):
                if hid == 4:
                    x1 = int(lm.x * width)
                    y1 = int(lm.y * height)
                if hid == 8:
                    x2 = int(lm.x * width)
                    y2 = int(lm.y * height)
                    
    # Capture image by pressing index and thumb finger
    if x1 != 0 and x2 != 0:
        dist = (pow(y2 - y1,2) + pow(x2 - x1,2))**0.5
        if dist < 20:
            time.sleep(1)
            cv2.imwrite(f"capture{cou}.png",frame)
            cou += 1
            print("Image captured...")
        for lis in os.listdir():
            # checking for only .png files and removing all except the last
            if '.png' in lis:
                if f'capture{cou-1}.png' != lis:
                    os.remove(lis)
                        
    cv2.imshow("Video",frame)
    
    # Capture image by clicking button 'C' on keyboard
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite(f"capture{cou}.png",frame)
        cou += 1
        print("Image captured...")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()