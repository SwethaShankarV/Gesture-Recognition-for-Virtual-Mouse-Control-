import cv2
from HandTracker import HandDetector
import os
import numpy as np

# var
width, height = 1280, 720
folderpath = "review ppt"
# camera
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# list of ppt
pathimages = sorted(os.listdir(folderpath), key=len)
# print(pathimages)

# variables
imgnumber = 0
hs, ws = 120, 213
gesture_threshold = 400
button_press = False
button_counter = 0
button_delay = 20
annotations = [[]]
annotation_number = -1
annotation_start = False

# hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # importing images.
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathfullimage = os.path.join(folderpath, pathimages[imgnumber])

    # 1. confirm current working dir
    print("→ CWD:", os.getcwd())
    # 2. confirm slide folder exists
    print("→ slide‑folder exists?", os.path.isdir(folderpath))
    # 3. list what files we actually see in that folder
    print("→ files in folder:", os.listdir(folderpath))
    # 4. print the exact path we’re about to load
    pathfullimage = os.path.join(folderpath, pathimages[imgnumber])
    print("→ loading image:", pathfullimage, "→ exists?", os.path.isfile(pathfullimage))
    
    imgcurrent = cv2.imread(pathfullimage)

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gesture_threshold), (width, gesture_threshold), (0, 255, 0), 10)
    
    if hands and button_press == False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']

        lmlist = hand['lmList']

        # constraining window
        xVal = int(np.interp(lmlist[8][0], [width//2, width], [0, width]))
        yVal = int(np.interp(lmlist[8][1], [150, height-150], [0, height]))
        index_finger = xVal, yVal # index finger

        # print(fingers)
        if cy <= gesture_threshold: # hand should be above line
            # gesture 1 - left slide
            if fingers == [1, 0, 0, 0, 0]:
                print("left")
                if imgnumber > 0:
                    button_press = True
                    annotations = [[]]
                    annotation_number = -1
                    annotation_start = False
                    imgnumber -= 1
            # gesture 2 - right slide
            if fingers == [0, 0, 0, 0, 1]:
                print("right")
                if imgnumber < len(pathimages)-1:
                    button_press = True
                    annotations = [[]]
                    annotation_number = -1
                    annotation_start = False
                    imgnumber += 1

        # gesture 3 - Show pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgcurrent, index_finger, 12, (0, 0, 255), cv2.FILLED)
            # annotation_start = False

        # gesture 4 - draw
        if fingers == [0, 1, 0, 0, 0]:
            if annotation_start is False:
                annotation_start = True
                annotation_number += 1
                annotations.append([])
            annotations[annotation_number].append(index_finger)
            cv2.circle(imgcurrent, index_finger, 12, (0, 0, 255), cv2.FILLED)

        else:
            annotation_start = False
        
        # Gesture 5 - erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotation_number -= 1
                button_press = True

    else:
        annotation_start = False

    # button iter
    if button_press:
        button_counter += 1
        if button_counter > button_delay:
            button_counter = 0
            button_press = False

    # Drawing part
    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgcurrent, annotations[i][j-1], annotations[i][j], (0, 0, 200), 12)

    # adding webcam image on the slides.
    imgsmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgcurrent.shape
    imgcurrent[0:hs, w-ws:w] = imgsmall

    cv2.imshow("slides", imgcurrent)
    cv2.imshow('Input', img)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()