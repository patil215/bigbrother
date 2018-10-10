import cv2
import imutils
import pickle
import argparse
import sys
from fileutils import read_obj, write_obj

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dest", required=True)
ap.add_argument("-v", "--video", required=True)
args = vars(ap.parse_args())

VIDEO_SOURCE = cv2.VideoCapture(args["video"])

startIndex = 0
endIndex = 0
currentIndex = -1

videoFrames = []
while True:
    currentIndex += 1
    rawFrame = VIDEO_SOURCE.read()[1]
    if rawFrame is None:
        break
    
    videoFrames.append(rawFrame)
    croppedFrame = imutils.resize(rawFrame, width=1600)
    cv2.imshow("preview", croppedFrame)

    key = cv2.waitKey(0) & 0xFF
    if key == ord("s"):
        startIndex = currentIndex
    elif key == ord("e"):
        endIndex == currentIndex
        break
    elif key == ord("n"):
        continue
    elif key == ord("q"):
        sys.exit(1)

trim = videoFrames[startIndex:endIndex]
write_obj(args["dest"], trim)
