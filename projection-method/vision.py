import numpy as np
import cv2
import imutils

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

def getTracePathUsingColor(vs, lower_bound=(55, 82, 90), upper_bound=(96, 255, 255)):
	""" Given a video of a pen drawing a single digit, gets a TracePath object associated with its trajectory.
	vs - cv2.VideoCapture object
	This specific method attempts to do so by using the HSV color space of the tip of the pen."""
	path = TracePath()

	while True:
		frame = vs.read()[1]
		if frame is None:
			break

		timestamp = v.get(cv2.CAP_PROP_POS_MSEC)
		print(timestamp)

		# Indiscriminately add the first point
		greenLocs = findColor(frame, lower_bound, upper_bound)
		point = getPts(greenLocs)[0]
		path.add((point[0], point[1], 0), timestamp)

	return path