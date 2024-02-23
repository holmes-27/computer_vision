import cv2

img = cv2.imread(r"Photos/shapes.png")
# Image copy is made to display the contours and text
imgCpy = img.copy()

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv2.drawContours(imgCpy, cnt, -1, (0,0,255),2)
        peri = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, closed=True)
        
        if len(approx) == 3:
            txt = "triangle"
        elif len(approx) == 4:
            txt = "quadrilateral"
        elif len(approx) == 5:
            txt = "pentagon"
        elif len(approx) == 6:
            txt = "hexagon"
        else:
            txt = "circle"
            
        x,y,w,h = cv2.boundingRect(approx)
        cv2.putText(imgCpy, txt, (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0),2)
        
        
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0.5)
imgCanny = cv2.Canny(imgBlur, 50, 50)

getContours(imgCanny)

cv2.imshow("Original Image", img)
cv2.imshow("Canny Image",imgCanny)
cv2.imshow("Output",imgCpy)

cv2.waitKey(0)
cv2.destroyAllWindows()