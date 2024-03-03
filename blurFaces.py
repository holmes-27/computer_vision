import cv2

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

'''
# Create trackbar
cv2.namedWindow("trackbar")

def nothing(x):
    pass

cv2.createTrackbar("kernel", "trackbar", 1, 50, nothing)
cv2.createTrackbar("sigma", "trackbar", 1, 50, nothing)
'''

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray,1.1,3)
    
    for x,y,w,h in faces:
        ''' 
        # To draw the boundary for face 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        
        # Get kernel and sigma values from trackbar to blur faces 
        k = cv2.getTrackbarPos("kernel", "trackbar")
        sig = cv2.getTrackbarPos("sigma", "trackbar")
        '''
        
        k = 30
        sig = 20
        
        # threshold to blur the whole face
        th1,th2 = 40,10
        
        face = frame[y-th1:y+h,x-th2:x+h+th2]
        frame[y-th1:y+h,x-th2:x+h+th2] = cv2.GaussianBlur(face, (2*k+1,2*k+1), sig)
    
    cv2.imshow("Video",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()