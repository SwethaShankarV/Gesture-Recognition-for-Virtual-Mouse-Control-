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
    img = cv2.flip(img, 1) # mirror
    pathfullimage = os.path.join(folderpath, pathimages[imgnumber])

    # 1. confirm current working dir
    print("→ CWD:", os.getcwd())
    # 2. confirm slide folder exists
    print("→ slide-folder exists?", os.path.isdir(folderpath))
    # 3. list what files we actually see in that folder
    print("→ files in folder:", os.listdir(folderpath))
    # 4. print the exact path we’re about to load
    pathfullimage = os.path.join(folderpath, pathimages[imgnumber])
    print("→ loading image:", pathfullimage, "→ exists?", os.path.isfile(pathfullimage))
    
    imgcurrent = cv2.imread(pathfullimage) # read the current slide

    hands, img = detector.findHands(img) # Hand  and frame (img) detection
    cv2.line(img, (0, gesture_threshold), (width, gesture_threshold), (0, 255, 0), 10) # draw line in (0, 255, 0) colour and 10 size
    
    if hands and button_press == False: # if hand is detected and no recent swipe
        hand = hands[0] # only first hand is required
        fingers = detector.fingersUp(hand) # Detect open/up fingers
        cx, cy = hand['center'] # Hand center x, y

        lmlist = hand['lmList'] # List of all landmarks

        # constraining window
        xVal = int(np.interp(lmlist[8][0], [width//2, width], [0, width])) #map the position of the index finger from a specific region 
        #within a larger frame or image to a new coordinate system representing a constrained window or control area
        yVal = int(np.interp(lmlist[8][1], [150, height-150], [0, height]))
        index_finger = xVal, yVal # index finger

        # print(fingers)
        if cy <= gesture_threshold: # hand should be above line

            # gesture 1 - left slide
            if fingers == [1, 0, 0, 0, 0]:
                print("left")
                if imgnumber > 0: # This checks if the current image number is greater than 0, meaning there are previous images to navigate to
                    button_press = True # A flag to signal that a button has been "pressed" virtually due to the gesture
                    annotations = [[]] # When a slide is navigated, any existing annotations(drawings/scribbled notes) are cleared and the annotation state is reset
                    annotation_number = -1
                    annotation_start = False
                    imgnumber -= 1

            # gesture 2 - right slide
            if fingers == [0, 0, 0, 0, 1]:
                print("right")
                if imgnumber < len(pathimages)-1: # This checks if the current image number is lesser than last image
                    button_press = True
                    annotations = [[]]
                    annotation_number = -1
                    annotation_start = False
                    imgnumber += 1 # This increments imgnumber, moving to the next image in the carousel

        # gesture 3 - Show pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgcurrent, index_finger, 12, (0, 0, 255), cv2.FILLED) # If the gesture is detected, a red circle is drawn on the imgcurrent 
            # image at the index_finger position

        # gesture 4 - draw
        if fingers == [0, 1, 0, 0, 0]:
            if annotation_start is False: # checks if the drawing mode is currently inactive
                annotation_start = True # drawing mode is activated
                annotation_number += 1
                annotations.append([]) # A new empty list is appended to annotations to store the coordinates of the current drawing stroke
            annotations[annotation_number].append(index_finger) # The current position of the index finger is added to the current annotation path, recording the drawing motion
            cv2.circle(imgcurrent, index_finger, 12, (0, 0, 255), cv2.FILLED) # A red circle is drawn on the imgcurrent at the index_finger position

        else:
            annotation_start = False # If any other gesture is made (or no hand is detected), the drawing mode is turned off
        
        # Gesture 5 - erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations: # This checks if there are any annotations present on the screen to be erased
                annotations.pop(-1) # removes the last element from the annotations list. Since each element
                # in annotations represents a complete drawing stroke or path, pop(-1) erases the most recently drawn annotation
                annotation_number -= 1
                button_press = True # for controlling the flow of the application or preventing immediate re-triggering of the same action

    else:
        annotation_start = False

    # button iter
    if button_press:
        button_counter += 1
        if button_counter > button_delay:
            button_counter = 0 # The counter is reset if the threshold (button_delay) is passed
            button_press = False # The button_press flag is set to False, enabling recognition of the next gesture

    # Drawing part
    for i in range(len(annotations)): # A loop iterates through each complete annotation stroke stored in the annotations list
        for j in range(len(annotations[i])): # A nested loop iterates through each point within a specific annotation stroke
            if j != 0:
                cv2.line(imgcurrent, annotations[i][j-1], annotations[i][j], (0, 0, 200), 12) # A line is drawn between consecutive 
                # points of the current annotation stroke on the imgcurrent image

    # adding webcam image on the slides.
    imgsmall = cv2.resize(img, (ws, hs)) # The current webcam frame (img) is resized to a smaller size
    h, w, _ = imgcurrent.shape # This displays the raw webcam feed (img) in a window named "Input"
    imgcurrent[0:hs, w-ws:w] = imgsmall # takes the resized webcam image (imgsmall) and directly places it onto the top-right corner 
    # of the current presentation slide image (imgcurrent), creating a Picture-in-Picture (PiP) effect

    cv2.imshow("slides", imgcurrent) # This displays the current slide with annotations
    cv2.imshow('Input', img)

    c = cv2.waitKey(1) # This waits for a key press for 1 millisecond
    if c == 27: # This checks if the Escape key (ASCII code 27) was pressed
        break

cap.release() # This releases the webcam resource
cv2.destroyAllWindows() # This closes all OpenCV windows