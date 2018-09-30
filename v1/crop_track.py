from collections import deque
from imutils.video import VideoStream
from drawPath import drawPath
from createBlank import create_blank
import numpy as np
import argparse
import cv2
import imutils
import time
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

greenLower = (55, 82, 90)
greenUpper = (96, 255, 255)
#orangeLower = (8, 124, 178)
#orangeUpper = (255, 255, 255)

orangeLower = (6, 120, 170)
orangeUpper = (255, 255, 255)

if not args.get("video", False):
        print("Please supply video file")
        sys.exit(0)
else:
        vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)

def findColor(frame, min_bound, max_bound):
        BLUR_RADIUS = 11
        blurred = cv2.GaussianBlur(frame, (BLUR_RADIUS, BLUR_RADIUS), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, min_bound, max_bound)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        return cnts

def getPts(cnts):
        return [cv2.minEnclosingCircle(c)[0] for c in cnts]

def drawColorLocations(frame, cnts):
        for c in cnts:
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

greenPointsTrail = []
greenPointsWindow = "green"
orangePointsTrail = []
orangePointsWindow = "orange"
mainWindow = "frame"
frameScaleWidth = 1600
cv2.namedWindow(mainWindow)

frame = vs.read()[1]
frame = imutils.resize(frame, width=frameScaleWidth)
if frame is None:
        vs.release()
        sys.exit(1)

refPt = []
def click_and_crop(event, x, y, flags, param):
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))

        # draw a rectangle around the region of interest
        frameCopy = frame.copy()
        cv2.rectangle(frameCopy, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow(mainWindow, frameCopy)

cv2.setMouseCallback(mainWindow, click_and_crop)

while True:
        cv2.imshow(mainWindow, frame)
        key = cv2.waitKey(0)

        if key == ord('q'):
                sys.exit(1)

        if key == ord('c') and len(refPt) == 2:
                break

cv2.setMouseCallback(mainWindow, lambda *args : None)

while True:
        rawFrame = vs.read()[1]

        if rawFrame is None:
                break

        rawFrame = imutils.resize(rawFrame, width=frameScaleWidth)
        frame = rawFrame.copy()[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow(mainWindow, frame)

        greenLocs = findColor(frame, greenLower, greenUpper)
        greenPoints = getPts(greenLocs)
        drawColorLocations(frame, greenLocs)
        greenPointsTrail += greenPoints

        orangeLocs = findColor(frame, orangeLower, orangeUpper)
        orangePoints = getPts(orangeLocs)
        drawColorLocations(frame, orangeLocs)
        orangePointsTrail += orangePoints

        drawPath(greenPointsWindow, greenPointsTrail, 10.0)
        drawPath(orangePointsWindow, orangePointsTrail, 10.0)
 
        # show the frame to our screen
        cv2.imshow(mainWindow, frame)
        key = cv2.waitKey(1) & 0xFF
 
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
                break


vs.release()
 
# close all windows
cv2.destroyAllWindows()

