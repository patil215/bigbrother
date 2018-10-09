import cv2
import imutils
import math

def findColor(frame, min_bound, max_bound):
    BLUR_RADIUS = 11
    blurred = cv2.GaussianBlur(frame, (BLUR_RADIUS, BLUR_RADIUS), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, min_bound, max_bound)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    cnts = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    return cnts

def distance(pointA, pointB):
    return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

def trackColor(frame, path, color):
    locations = findColor(frame, color[0], color[1])
    if len(locations) == 0:
        return None

    firstPoint = locations[0]
    centerPoint = getCenterPoints([firstPoint])[0]

    if len(path) == 0 or distance(path[-1], centerPoint) < 100:
        path.append(centerPoint)

    return firstPoint

def getCenterPoints(cnts):
    return [cv2.minEnclosingCircle(c)[0] for c in cnts]

