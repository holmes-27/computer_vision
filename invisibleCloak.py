import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# capture a background image without shaking
bgImg = cv2.imread("background.png")

while cap.isOpened():
    ret, frame = cap.read()
    
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower1 = np.array([0,120,70])
    upper1 = np.array([10,255,255])
    mask1 = cv2.inRange(frameHSV, lower1, upper1)
    
    lower2 = np.array([120,120,70])
    upper2 = np.array([180,255,255])
    mask2 = cv2.inRange(frameHSV, lower2, upper2)
    
    mask = mask1 + mask2
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=10)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8),iterations=10)
    out1 = cv2.bitwise_and(bgImg, bgImg,mask=mask)
    
    maskNot = cv2.bitwise_not(mask)    
    out2 = cv2.bitwise_and(frame, frame,mask=maskNot)
    
    output = out1 + out2
    
    
    cv2.imshow("Invisbility Cloak",output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()