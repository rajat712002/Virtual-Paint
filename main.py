import cv2 as cv
import numpy as np
import time
import math
import hand_tacking_module as htm

cWidth = 640
cHeight = 360
cTime = 0
pTime = 0
colors = {'red' :(0,0,255),'green':(0,255,0), 'blue':(255,0,0), 'eraser':(0,0,0)}
imgNames = ['static', 'eraser', 'red','green', 'blue']
paintImgs = {}
thickness = 10
detector = htm.HandTracker()
selected = 'static'
pt1 = (int(cWidth * 0.95), int(cHeight * 0.9))
pt2 = (int(cWidth * 0.94), int(cHeight * 0.4))
pt3 = (int(cWidth * 0.96), int(cHeight * 0.4))
sliderLength = pt1[1] - pt2[1]
sliderCircle = (int(cWidth*0.95), int(cHeight*0.45))

for name in imgNames:
    img = cv.imread(f'resources/{name}.png')
    img = cv.resize(img, (cWidth, cHeight))
    paintImgs[name] = img

def calDist(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    distance = math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))
    return distance

canvas = np.zeros((cHeight, cWidth, 3), np.uint8)

cam = cv.VideoCapture(0)
cam.set(3, cWidth)
cam.set(4, cHeight)

while 1:
    success, frame = cam.read()
    frame = cv.flip(frame, 1)

    detector.findHands(frame, False)
    positions = detector.findPosition(frame)

    if len(positions):
        finger1 = (positions[8][1], positions[8][2])
        finger2 = (positions[12][1], positions[12][2])
        dist = calDist(finger1, finger2)

        if (dist<25) :
             xr = finger1[0]/cWidth
             yr = finger1[1]/cHeight

             if 0.14<=yr<=0.22:
                 if (0.19<=xr<=0.27): selected = 'eraser'
                 if (0.44<=xr<=0.51): selected = 'red'
                 if (0.60<=xr<=0.68): selected = 'green'
                 if (0.76<=xr<=0.86): selected = 'blue'

             elif 0.4<=yr<=0.9 and 0.94<=xr<=0.96:
                 dist = 0.9*cHeight - finger1[1]
                 thickness = int((dist / sliderLength) * sliderLength / 2)
                 thickness = int((dist / sliderLength) * sliderLength / 4)
                 sliderCircle = (sliderCircle[0], finger1[1])

        else :
            if selected == 'eraser':
                cv.circle(canvas, finger1, thickness, colors[selected], -1)
            elif selected != 'static':
                cv.circle(canvas, finger1, thickness, colors[selected], -1)

        if selected == 'eraser':
            cv.circle(frame, finger1, thickness, (255, 255, 255), 2)
        elif selected != 'static':
            cv.circle(frame, finger1, thickness, colors[selected], -1)
            cv.circle(frame, finger1, thickness + 2, (255, 255, 255), 2)
    else :
        cv.putText(frame, 'No hand detected', (100, cHeight-30), cv.FONT_HERSHEY_PLAIN, 2, (0,100,255), 2)

    imgGray = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 40, 255,cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    frame = cv.bitwise_and(frame, imgInv)
    frame = cv.bitwise_or(frame, canvas)

    cTime = time.time()
    fps = int(1/(cTime - pTime))
    pTime = cTime
    cv.putText(frame, str(fps), (10,cHeight-30), cv.FONT_HERSHEY_PLAIN, 2, (80,255,20), 2)

    frame[:90, :] = paintImgs[selected][:90, :]

    triangle_cnt = np.array([pt1, pt2, pt3])
    cv.drawContours(frame, [triangle_cnt], 0, (255, 255, 255), -1)
    cv.circle(frame, sliderCircle, 10, (0, 0, 0), -1)
    cv.circle(frame, sliderCircle, 8, (255, 255, 255), -1)

    cv.imshow('camera', frame)

    cv.waitKey(1)