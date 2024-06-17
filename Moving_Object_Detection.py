import imutils
import time
import cv2

cv = cv2.VideoCapture(0)
firstframe = None
area = 500

while True:
    _ , img = cv.read()
    text = "NOTHING"
    img = imutils.resize(img,width=500)
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grayimg = cv2.GaussianBlur(grayimg,(21,21),0)
    if firstframe is None:
        firstframe = grayimg
        continue
    imgdiff = cv2.absdiff(firstframe,grayimg)
    threshimg = cv2.threshold(imgdiff,50,255,cv2.THRESH_BINARY)[1]
    threshimg = cv2.dilate(threshimg,None,iterations=2)
    cnts = cv2.findContours(threshimg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c)<area:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img , (x,y), (x+w,y+h),(0,255,0),2)
        text="MOVING OBJECT DETECTED"
        print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow("lokesh",img)
    '''cv2.imshow("lokesh",threshimg)
    cv2.imshow("lokesh",imgdiff)'''
    key = cv2.waitKey(1) & 0xFF
    if key == ord("k"):
        break
cv.release()
cv2.destroyAllWindows()