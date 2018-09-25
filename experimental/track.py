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

def drawPath(frame, ps):
	pts = [(int(pt[0] * 1.5) - 500, int(pt[1] * 1.5) - 500) for pt in ps]
	for index in range(len(pts) - 1):
		cv2.line(frame, pts[index], pts[index + 1], (255, 255, 255))


points = []
points2 = []
while True:
	frame = vs.read()[1]

	if frame is None:
		break

	frame = imutils.resize(frame, width=1600)

	greenLocs = findColor(frame, greenLower, greenUpper)
	greenPoints = getPts(greenLocs)
	drawColorLocations(frame, greenLocs)
	points2 += greenPoints
	drawPath(frame, points2)

	orangeLocs = findColor(frame, orangeLower, orangeUpper)
	orangePoints = getPts(orangeLocs)
	drawColorLocations(frame, orangeLocs)

	points += orangePoints
	drawPath(frame, points)
 
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

 
vs.release()
 
# close all windows
cv2.destroyAllWindows()

