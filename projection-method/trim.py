import cv2
import imutils
import pickle
import sys
from fileutils import read_obj, write_obj
import click
from readvideo import getTracePathFromFrames
import os
import easygui
import sys

@click.command()
@click.argument("video")
@click.option("-d", "--dest", default="data", required=False, help="Destination folder for segments and paths")
@click.option("-t", "--trace", is_flag=True, help="Enable trace path recording mode")
def segment(video, dest, trace):
    VIDEO_SOURCE = cv2.VideoCapture(video)

    if not os.path.exists(dest):
        os.makedirs(dest)

    frameIndex = 0
    clipIndex = 0

    while True:
        startIndex = -1
        while True:
            VIDEO_SOURCE.set(1, frameIndex)
            ok, rawFrame = VIDEO_SOURCE.read()
            if not ok:
                sys.exit(1)

            frame = imutils.resize(rawFrame, height=700)
            cv2.imshow("Clipper", frame)

            key = cv2.waitKey(0) & 0xFF
            if key == ord("s"):
                videoFrames = []
                startIndex = frameIndex
                print("Starting at frame {0}".format(frameIndex))
                continue
            elif key == ord("e"):
                print("Ending at frame {0}".format(frameIndex))
                for index in range(startIndex, frameIndex + 1):
                    VIDEO_SOURCE.set(1, index)
                    ok, rawFrame = VIDEO_SOURCE.read()
                    videoFrames.append(rawFrame)
                break
            elif key == ord("b"):
                frameIndex = frameIndex - 1 if frameIndex > 0 else frameIndex
                continue
            elif key == ord("n"):
                frameIndex += 1
                continue
            elif key == ord("f"):
                frameIndex += 20
                continue
            elif key == ord("a"):
                frameIndex = frameIndex - 20 if frameIndex > 20 else frameIndex
                continue
            elif key == ord("q"):
                sys.exit(1)

        print(len(videoFrames))
        # Prompt the user for the class to store the segment as
        class_name = easygui.enterbox("What is the class of this data? (zero, eight, etc)")
        segment_dir = dest + '/segments/' + class_name
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir)
        print("Saving segment...")
        write_obj(segment_dir + '/' + str(clipIndex), videoFrames)

        if trace:
            path = getTracePathFromFrames(videoFrames)
            path_dir = dest + '/paths/' + class_name
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            print("Saving path")
            write_obj(path_dir + '/' + str(clipIndex), path)

        clipIndex += 1


if __name__ == '__main__':
    segment()