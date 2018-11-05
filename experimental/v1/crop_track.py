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

# define color ranges for tracking
GREEN_COLOR_LOWER_BOUND = (55, 82, 90)
GREEN_COLOR_UPPER_BOUND = (96, 255, 255)

ORANGE_COLOR_LOWER_BOUND = (6, 120, 170)
ORANGE_COLOR_UPPER_BOUND = (255, 255, 255)

if not args.get("video", False):
        print("Please supply video file")
        sys.exit(0)
else:
        VIDEO_SOURCE = cv2.VideoCapture(args["video"])

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

# define window names
WINDOW_GREEN_PATH = "green"
WINDOW_ORANGE_PATH = "orange"
WINDOW_MAIN = "frame"
cv2.namedWindow(WINDOW_MAIN)

# scale factor for raw video
# 4K doesn't fit on the screen :(
FRAME_SCALE_WIDTH = 1600

greenPointsTrail = []
orangePointsTrail = []

# read the first frame
frame = VIDEO_SOURCE.read()[1]
frame = imutils.resize(frame, width=FRAME_SCALE_WIDTH)

if frame is None:
        VIDEO_SOURCE.release()
        sys.exit(1)

cropRefPt = []
def clickAndCrop(event, x, y, flags, param):
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    global cropRefPt
    if event == cv2.EVENT_LBUTTONDOWN:
        cropRefPt = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        cropRefPt.append((x, y))

        # draw a rectangle around the region of interest
        frameCopy = frame.copy()
        cv2.rectangle(frameCopy, cropRefPt[0], cropRefPt[1], (0, 255, 0), 2)
        cv2.imshow(WINDOW_MAIN, frameCopy)

paperRefPt = []
def click_and_transform(event, x, y, flags, param):
    global paperRefPt
    if event == cv2.EVENT_LBUTTONDOWN:
        paperRefPt.append((x, y))


cv2.setMouseCallback(WINDOW_MAIN, clickAndCrop)

while True:
        cv2.imshow(WINDOW_MAIN, frame)
        key = cv2.waitKey(0)

        if key == ord('q'):
                sys.exit(1)

        if key == ord('c') and len(cropRefPt) == 2:
                break

cv2.setMouseCallback(WINDOW_MAIN, click_and_transform)

frame = frame.copy()[cropRefPt[0][1]:cropRefPt[1][1], cropRefPt[0][0]:cropRefPt[1][0]]
rows, cols = frame.shape[:2]

while True:
        cv2.imshow(WINDOW_MAIN, frame)
        key = cv2.waitKey(0)

        if key == ord('p') and len(paperRefPt) == 4:
            break

src_pts = np.float32([paperRefPt[0], paperRefPt[1], paperRefPt[3], paperRefPt[2]])
dst_pts = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

print(frame)
print(projective_matrix)
sys.exit(0)

cv2.setMouseCallback(WINDOW_MAIN, lambda *args : None)

while True:
        rawFrame = VIDEO_SOURCE.read()[1]


        if rawFrame is None:
                break

        rawFrame = imutils.resize(rawFrame, width=FRAME_SCALE_WIDTH)
        frame = rawFrame.copy()[cropRefPt[0][1]:cropRefPt[1][1], cropRefPt[0][0]:cropRefPt[1][0]]

        frame = cv2.warpPerspective(frame.copy(), projective_matrix, frame.shape[:2])

        cv2.imshow(WINDOW_MAIN, frame)


        greenLocs = findColor(frame, GREEN_COLOR_LOWER_BOUND, GREEN_COLOR_UPPER_BOUND)
        greenPoints = getPts(greenLocs)
        drawColorLocations(frame, greenLocs)
        greenPointsTrail += greenPoints

        orangeLocs = findColor(frame, ORANGE_COLOR_LOWER_BOUND, ORANGE_COLOR_UPPER_BOUND)
        orangePoints = getPts(orangeLocs)
        drawColorLocations(frame, orangeLocs)
        orangePointsTrail += orangePoints

        drawPath(WINDOW_GREEN_PATH, greenPointsTrail, 10.0)
        drawPath(WINDOW_ORANGE_PATH, orangePointsTrail, 10.0)
 
        # show the frame to our screen
        cv2.imshow(WINDOW_MAIN, frame)
        key = cv2.waitKey(1) & 0xFF
 
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
                break


VIDEO_SOURCE.release()
 
# close all windows
cv2.destroyAllWindows()

