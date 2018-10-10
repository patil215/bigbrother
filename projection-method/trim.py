import cv2
import imutils
import pickle
import argparse
import sys
from fileutils import read_obj, write_obj

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dest", required=True)
ap.add_argument("-v", "--video", required=True)
ap.add_argument("-s", "--start", required=False)
args = vars(ap.parse_args())

VIDEO_SOURCE = cv2.VideoCapture(args["video"])

savedIndex = 0
if args["start"] != None:
    savedIndex = int(args["start"])

startIndex = 0
endIndex = 0
currentIndex = 0
videoFrames = []

while True:
    while True:
        print("Current frame: {0}".format(currentIndex))
        if currentIndex >= len(videoFrames):
            rawFrame = VIDEO_SOURCE.read()[1]
            if rawFrame is None:
                break
            
            videoFrames.append(rawFrame)

        rawFrame = videoFrames[currentIndex]
        croppedFrame = imutils.resize(rawFrame, width=1600)
        cv2.imshow("preview", croppedFrame)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("s"):
            startIndex = currentIndex
            print("Set start, end at frame: [{0}, {1}]".format(startIndex, endIndex))
            continue
        elif key == ord("e"):
            endIndex = currentIndex
            print("Set start, end at frame: [{0}, {1}]".format(startIndex, endIndex))
            break
        elif key == ord("b"):
            if currentIndex > 0:
                currentIndex -= 1
            continue
        elif key == ord("n"):
            currentIndex += 1
            continue
        elif key == ord("q"):
            sys.exit(1)

    filename = args["dest"] + "{0}".format(savedIndex)
    print("Saving range [{0}, {1}] ...".format(startIndex, endIndex, filename))

    # python excludes the last index, so we add 1 to it to get the endIndex frame too
    trim = videoFrames[startIndex:endIndex + 1]
    write_obj(filename, trim)
    print("Saved to {2}".format(startIndex, endIndex, filename))

    startIndex = currentIndex
    endIndex = 0
    savedIndex += 1
    print("Set start, end at frame [{0}, {1}]".format(startIndex, endIndex))
