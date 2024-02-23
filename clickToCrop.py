import cv2
import numpy as np

img = cv2.imread(r"Photos/cards.png")
img = cv2.resize(img,(500,500))

pts = []
def mouseEvent(event,x,y,flags,params):
    global pts
    if cv2.EVENT_LBUTTONDOWN == event:
        pts.append([x,y])
                
height,width,_ = img.shape
        
# Get the points of the image to be retrieved using mouse click
while True:
    cv2.imshow("Image",img)
    # cv2.imshow(winname1), cv2.setMouseCallback(winname2), winname1 = winname2
    cv2.setMouseCallback("Image", mouseEvent)
    # To perform the warpPerspective pts1 should have 4 points where 'pts' list is empty
    if len(pts) == 4:
        pts1 = np.float32([pts[0],pts[1],pts[2],pts[3]])
        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        output = cv2.warpPerspective(img, matrix, (width,height))
                
        cv2.imshow("Output",output)
        
        # Close the output and click anywhere again
        pts = []
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()