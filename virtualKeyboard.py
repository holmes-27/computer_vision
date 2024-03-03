import cv2
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import mediapipe as mp

# calculate the distance between the 2 fingers to perform click operation
def eucDist(x1,y1,x2,y2):
    return (pow(x2-x1,2) + pow(y2-y1,2))**0.5

# drawing keyboard of 3 layers for letters and 1 layer for spacebar and backspace
def drawKeyboard(img):
    c = img.shape[1]//10
    fontColor = (0,0,0)
    boardColor = (0,0,255)
    boardThickness = cv2.FILLED

    text = list('qwertyuiop')
    x1,y1,x2,y2 = 0,0,c,c
    pts = []
    for i in range(10):
        pts.append([text[i],x1,y1,x2,y2]) 
        cv2.rectangle(img,(x1,y1),(x2,y2),boardColor,boardThickness)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),5)
        mid = int((x1 + x2)/2)-15,int((y1 + y2)/2)+5
        cv2.putText(img, text[i], mid, cv2.FONT_HERSHEY_COMPLEX, 1.5, fontColor,2)
        x1 = x2
        x2 = c*(i+2)
    
    text = list('asdfghjkl')
    x1,y1,x2,y2 = c//2,c,c+c//2,c*2
    for i in range(9):
        pts.append([text[i],x1,y1,x2,y2]) 
        cv2.rectangle(img,(x1,y1),(x2,y2),boardColor,boardThickness)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),5)
        mid = int((x1 + x2)/2)-15,int((y1 + y2)/2)+5
        cv2.putText(img, text[i], mid, cv2.FONT_HERSHEY_COMPLEX, 1.5, fontColor,2)
        x1 += c
        x2 += c    
    
    text = list('zxcvbnm')
    x1,y1,x2,y2 = c+c//2,c*2,2*c+c//2,c*3
    for i in range(7):
        pts.append([text[i],x1,y1,x2,y2])    
        cv2.rectangle(img,(x1,y1),(x2,y2),boardColor,boardThickness)
        # board outline 
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),5)
        mid = int((x1 + x2)/2)-20,int((y1 + y2)/2)+5
        cv2.putText(img, text[i], mid, cv2.FONT_HERSHEY_COMPLEX, 1.5, fontColor,2)
        x1 += c
        x2 += c
    
    # spacebar
    x1,y1,x2,y2 = c+c//2,c*3,5*c+c//2,c*4-20
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),boardThickness)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),5)
    mid = int((x1 + x2)/2)-50,int((y1 + y2)/2)+5
    cv2.putText(img, 'space', mid, cv2.FONT_HERSHEY_COMPLEX, 1, fontColor,2)
    pts.append([" ",x1,y1,x2,y2])
    
    # backspace
    x1,y1,x2,y2 = 5*c+c//2,c*3,8*c+c//2,c*4-20
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),boardThickness)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),5)
    mid = int((x1 + x2)/2)-80,int((y1 + y2)/2)+5
    cv2.putText(img, 'backspace', mid, cv2.FONT_HERSHEY_COMPLEX, 1, fontColor,2)
    pts.append(["clear",x1,y1,x2,y2])
    
    return pts


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(model_complexity=0,max_num_hands=1,min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

x8,x12 = 0,0
ptColor = (0,255,0)

# coordinates to display the text
textX,textY = 30,400

# main initialization
'''
* check -> Checks if a key is pressed, whether to display anything.
* res -> Adds the letter/space/backspace based on the key press.
* click -> Since the code is running in an infinite loop if a key is pressed there will be duplicates formed in the output. To overcome this, 'click' threshold is used, it gets incremented when there is a click operation performed which tells us how long the key is pressed. When 'click' values were printed the most recurring value was 21, unique values were 9 & 13. So, if 'cick' < 14, key is pressed.
* textList -> Contains 'res' values which includes letter/space/backspace, then it is joined which is the "output"
* output -> Displays the final text in the screen.
'''
check = False
click = 0
res = ''
textList = []
output = ''

# main code
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(800,512))
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape
    
    # returns [letter/space/clear, x1, y1, x2, y2]
    lis = drawKeyboard(frame)
    
    result = hands.process(frameRGB)
    
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand,mpHands.HAND_CONNECTIONS)
            for hid,lm in enumerate(hand.landmark):
                # index finger
                if hid == 8:
                    x8 = int(lm.x * width)
                    y8 = int(lm.y * height)
                    #cv2.circle(frame,(x8,y8),10,ptColor,cv2.FILLED)
                    
                # middle finger
                if hid == 12:
                    x12 = int(lm.x * width)
                    y12 = int(lm.y * height)
                    #cv2.circle(frame,(x12,y12),10,ptColor,cv2.FILLED)
                
                if x8 != 0 and x12 != 0:
                    #cv2.line(frame,(x8,y8),(x12,y12),ptColor,5)
                    
                    # mid points
                    mp1,mp2 = (x8 + x12)//2,(y8 + y12)//2
                    #cv2.circle(frame,(mp1,mp2),10,(255,0,0),cv2.FILLED)
                    
                    res = ''
                    for txt,ptx1,pty1,ptx2,pty2 in lis:
                        # checks the hands are inside the keyboard area based on the mid points
                        if ptx1 < mp1 < ptx2 and pty1 < mp2 < pty2:
                            # color change while hovering over the keys
                            cv2.rectangle(frame,(ptx1,pty1),(ptx2,pty2),(0,0,128),cv2.FILLED)
                            
                            '''
                            while hovering "space", "backspace" and alphabets will become hidden. 
                            To overcome this, the text is rewritten. 
                            '''
                            # for space
                            if txt == " ":
                                mid = int((ptx1 + ptx2)/2)-50,int((pty1 + pty2)/2)+5
                                cv2.putText(frame, 'space', mid, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)
                            # for backspace
                            elif txt == "clear":
                                mid = int((ptx1 + ptx2)/2)-80,int((pty1 + pty2)/2)+5
                                cv2.putText(frame, 'backspace', mid, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)
                            # for alphabets
                            else:
                                mid = int((ptx1 + ptx2)/2)-15,int((pty1 + pty2)/2)+5
                                cv2.putText(frame, txt, mid, cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0),2)
                            
                            # checking the distance between 2 fingers to perform click 
                            if eucDist(x8, y8, x12, y12) < 30:           
                                if txt == 'clear':
                                    #cv2.circle(frame,(mp1,mp2),10,(0,255,255),cv2.FILLED)
                                    
                                    # when clicked orange color is displayed
                                    cv2.rectangle(frame,(ptx1,pty1),(ptx2,pty2),(0,165,255),cv2.FILLED)
                                    check = True
                                    click += 1
                                    res = txt
                                else:
                                    #cv2.circle(frame,(mp1,mp2),10,(0,255,255),cv2.FILLED)
                                    
                                    # when clicked orange color is displayed
                                    cv2.rectangle(frame,(ptx1,pty1),(ptx2,pty2),(0,165,255),cv2.FILLED)
                                    check = True
                                    click += 1
                                    res += txt
                                # if 'mid' coordinates match the 'txt' coordinates area
                                # the corresponding letter from 'lis' is added, 
                                # which doesn't require complete iteration for 'lis'  
                                break
                            
    if check:
        if res != '' and click < 14:
            if res == "clear" and textList:
                textList.pop()
            else:
                textList.append(res)
                
            output = ''.join(textList)
            
        click = 0
        check = False  
    
    # displays the text, '|' acts as cursor
    cv2.putText(frame, output+'|', (textX,textY), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255,255,0),2)                   
        
    cv2.imshow("Video",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()