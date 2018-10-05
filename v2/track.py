from collections import deque
from imutils.video import VideoStream
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

import cv2
import numpy as np

def create_blank(width, height, rgb_color=(0, 0, 0)):
        """Create new image(numpy array) filled with certain color in RGB"""
        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)

        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed(rgb_color))
        # Fill image with color
        image[:] = color

        return image

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

def drawPath(frameName, ps):
        pts = []
        for i in range(len(ps)):
                pt = ps[i]
                pts += [[int(pt[0] * 10.0), int(pt[1] * 10.0)]]

        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')

        for pt in pts:
                if pt[0] < min_x:
                        min_x = pt[0]
                if pt[1] < min_y:
                        min_y = pt[1]
                if pt[0] > max_x:
                        max_x = pt[0]
                if pt[1] > max_y:       
                        max_y = pt[1]

        for index in range(len(pts)):
                pts[index][0] = pts[index][0] - min_x
                pts[index][1] = pts[index][1] - min_y
                pts[index] = tuple(pts[index])

        max_x = max_x - min_x
        max_y = max_y - min_y

        #print("max_x: {0}, max_y {1}".format(max_x, max_y))
        if max_x <= 1 or max_y <= 1:
                return

        frame = create_blank(max_x, max_y)
        # print(pts)

        #pts = [(int(pt[0] * 3.0) - 1500, int(pt[1] * 3.0) - 1500) for pt in ps]
        for index in range(len(pts) - 1):
                cv2.line(frame, pts[index], pts[index + 1], (255, 255, 255))

        cv2.imshow(frameName, frame)

greenPointsTrail = []
greenPointsWindow = "green"
orangePointsTrail = []
orangePointsWindow = "orange"

while True:
        frame = vs.read()[1]

        if frame is None:
                break

        frame = imutils.resize(frame, width=1600)

        greenLocs = findColor(frame, greenLower, greenUpper)
        greenPoints = getPts(greenLocs)
        drawColorLocations(frame, greenLocs)
        greenPointsTrail += greenPoints

        orangeLocs = findColor(frame, orangeLower, orangeUpper)
        orangePoints = getPts(orangeLocs)
        drawColorLocations(frame, orangeLocs)
        orangePointsTrail += orangePoints

        drawPath(greenPointsWindow, greenPointsTrail)
        drawPath(orangePointsWindow, orangePointsTrail)
 
        # show the frame to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
                break


vs.release()
 
# close all windows
cv2.destroyAllWindows()

