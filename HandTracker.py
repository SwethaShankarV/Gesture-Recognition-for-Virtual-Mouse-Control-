import cv2
import mediapipe as mp
import math

class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower 
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode # If False (default), it uses tracking between frames
        self.maxHands = maxHands
        self.detectionCon = detectionCon # The minimum confidence value for the hand detection model to consider a detection successful
        self.minTrackCon = minTrackCon # The minimum confidence value for the hand tracking model to consider a hand tracked successfully
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon, min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils # Imports the drawing utilities for visualizing landmarks
        self.tipIds = [4, 8, 12, 16, 20] # Landmark IDs of the tips of the fingers
        self.fingers = [] #  An empty list to store the state of the fingers (e.g., whether they are extended or curled)
        self.lmList = [] # An empty list to store the pixel coordinates of the hand landmarks

    def findHands(self, img, draw=True, flipType=True): # for detecting hands in an image
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape # Get the height, width, and number of channels of the input image
        if self.results.multi_hand_landmarks: # if hands were detected in the image,
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks): # through each detected hand
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w) # Convert the normalized landmark coordinates (lm.x, lm.y) to pixel coordinates
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bounding box
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                    bbox[1] + (bbox[3] // 2) # Calculate the center coordinates of the bounding box
                
                myHand["lmList"] = mylmList # Store the list of landmark coordinates for the current hand
                myHand["bbox"] = bbox # Store the calculated bounding box
                myHand["center"] = (cx, cy) # Store the center of the bounding box

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"

                else: # If flipType is False, the hand type is taken directly from the MediaPipe results
                    myHand["type"] = handType.classification[0].label

                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) # Draws the 21 hand landmarks and their connections on the image
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 0, 255), 2) # rectangular box around hand
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2) # hand type label
        if draw:
            return allHands, img # both the allHands list (containing information about all detected hands) and the annotated image (img) are returned
        else:
            return allHands
        
    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        # A dictionary containing information about a single detected hand, 
        # including its type (Left or Right) and a list of its landmark coordinates (lmList)
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]: # if thumb tip's x-coordinate 
                    # is greater than the x-coordinate of the joint below it, that it, thumb is extended outward
                    fingers.append(1) # 1 is added to the fingers list, indicating the thumb is up
                else:
                    fingers.append(0)
            else: # for left hand opposite logic
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0) # thumb is closed/curled

            # 4 Fingers
            for id in range(1, 5): # compares the y-coordinate of the fingertip for remaining 4 fingers
                # If the y-coordinate of the fingertip is less than the y-coordinate of the MCP joint, the finger is considered extended (or "up")
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers
    
    def findDistance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
        Image with output drawn
        Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1) # Calculate Euclidean Distance
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None: # if an img (image) object is provided to the method
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED) # Draws a filled circle at p1 on the image of radius 15 pixels and colored purple
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # Draws a purple line connecting p1 and p2 on the image
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) # Draws a filled purple circle at the midpoint
            return length, info, img
        else:
            return length, info

def main():
        cap = cv2.VideoCapture(0)
        detector = HandDetector(detectionCon=0.8, maxHands=2)
        while True:
            # Get image frame
            success, img = cap.read()
            # Find the hand and its landmarks
            hands, img = detector.findHands(img) # with draw
            # hands = detector.findHands(img, draw=False) # without draw

            if hands:
                # Hand 1
                hand1 = hands[0]
                lmList1 = hand1["lmList"] # List of 21 Landmark point
                bbox1 = hand1["bbox"] # Bounding box info x,y,w,h
                centerPoint1 = hand1['center'] # center of the hand cx,cy
                handType1 = hand1["type"] # Handtype Left or Right
                
                fingers1 = detector.fingersUp(hand1)

                if len(hands) == 2:
                    # Hand 2
                    hand2 = hands[1]
                    lmList2 = hand2["lmList"] # List of 21 Landmark points
                    bbox2 = hand2["bbox"] # Bounding box info x,y,w,h
                    centerPoint2 = hand2['center'] # center of the hand cx,cy
                    handType2 = hand2["type"] # Hand Type "Left" or "Right"
                    fingers2 = detector.fingersUp(hand2)

                    # Find Distance between two Landmarks. Could be same hand or different hands
                    length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img) # with draw
                    # length, info = detector.findDistance(lmList1[8], lmList2[8]) # withdraw

                # Displays the img (with potential hand landmarks, bounding boxes, and distance lines drawn on it) in a window titled "Image"
                cv2.imshow("Image", img)
                cv2.waitKey(1)

if __name__ == "__main__":
    main()