import cv2 as cv
import mediapipe as mp

class HandTracker():
    def __init__(self, mode = False, maxHands = 2, cmplx=1, minDetCon=0.5, minTrackCon=0.5):
        self.ImageMode = mode
        self.nHands = maxHands
        self.complexity = cmplx
        self.minDetectionConfidence = minDetCon
        self.minTrackingConfidence = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.ImageMode, self.nHands, self.complexity,
                                        self.minDetectionConfidence, self.minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        frameRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.hands.process(frameRGB)
        if(self.result.multi_hand_landmarks):
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)


    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

        return lmList
