import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
waitTime=35 #time between each camera reading
offset = 20
imgSize = 300

classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["+","-","*","/","=","numbers"]

def countFingers():
    fingers1 = 0
    fingers2 = 0
    if hands:
        fingers1 = detector.fingersUp(hands[0]).count(1)
        if len(hands) == 2:     #case for two hands
            fingers2 = detector.fingersUp(hands[1]).count(1)
        else:
            fingers2 = 0
    else:
        fingers1 = 0

    return fingers1 + fingers2

def calculate(equation):
    if '+' in equation:
        y = equation.split('+')
        x = int(y[0])+int(y[1])
    elif '-' in equation:
        y = equation.split('-')
        x = int(y[0])-int(y[1])
    elif '*' in equation:
        y = equation.split('*')
        x = float(y[0])*int(y[1])
    elif '/' in equation:
        y = equation.split('/')
        print(y)
        x = float(y[0])/int(y[1])
    return x

previousChar=0
currentChar=0
counter=0
finalStr=""

recognizedSign=labels[0]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    imgOutput = img.copy()

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            try:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            except:
                continue

            if hand["type"]=="Right":               #only right hand can represent an arithmetic operation
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                if labels[index]=="numbers":
                    recognizedSign=str(countFingers())
                else:
                    recognizedSign=labels[index]
            else:
                recognizedSign=countFingers()
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            try:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            except:
                continue
            if hand["type"] == "Right":             #only right hand can represent an arithmetic operation
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                if labels[index] == "numbers":
                    recognizedSign = str(countFingers())
                else:
                    recognizedSign = labels[index]
            else:
                recognizedSign = countFingers()

        currentChar = recognizedSign
        if (currentChar != previousChar):
            previousChar = currentChar
            counter = 0
        else:
            counter += 1
            cv2.rectangle(img, (0, 450), (counter*20, 480), (0, 255, 0), -1)  #greeen bar indicator
            if (counter >= waitTime):           #time delay for each reading
                finalStr += str(recognizedSign)
                counter = 0
        if(recognizedSign=="="):
            try:
                finalStr = finalStr+'='+str(calculate(finalStr))
            except:
                continue

        cv2.rectangle(img, (0, 0), (750, 75), (29, 87, 20), -1)
        cv2.putText(img, finalStr, (50, 50), cv2.FONT_ITALIC, 1.7, (255, 255, 255), 3)

    else:
        finalStr=""     #clear screen when no hands in a frame


    cv2.imshow("Gesture Calculator", img)
    key=cv2.waitKey(1)
    if (key==27):
        cv2.destroyAllWindows()
        break