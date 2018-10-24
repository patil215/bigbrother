import cv2
import imutils
import pickle
import sys
from fileutils import read_obj, write_obj
import click
from readvideo import saveTracePathFromVideoSource
import os
import easygui
import sys
import threading
from collections import deque
from vizutils import draw_tracepoints

def save(video_source, startIndex, endIndex, filename):
    video = cv2.VideoCapture(video_source)
    videoFrames = []

    for index in range(startIndex, endIndex):
        video.set(1, index)
        ok, rawFrame = video.read()
        videoFrames.append(rawFrame)
    
    write_obj(filename, videoFrames)
    print("{0} frame segment saved successfully to {1}".format(len(videoFrames), filename))


@click.command()
@click.argument("video")
@click.option("-h", "--height", default=700, required=False, help="Video preview display height")
@click.option("-d", "--dest", default="data", required=False, help="Destination folder for segments and paths")
@click.option("-t", "--trace", is_flag=True, help="Enable trace path recording mode")
@click.option("-f", "--fps", default=60, required=False, help="Framerate of the input video, used for trace paths")
def segment(video, height, dest, trace, fps):
    VIDEO_SOURCE = cv2.VideoCapture(video)

    if not os.path.exists(dest):
        os.makedirs(dest)

    frameIndex = 0
    clipIndex = 0
    saveThreads = deque()

    while True:
        startIndex = -1
        while True:
            VIDEO_SOURCE.set(1, frameIndex)
            ok, rawFrame = VIDEO_SOURCE.read()
            if not ok:
                sys.exit(1)

            frame = imutils.resize(rawFrame, height=height)
            cv2.imshow("Clipper", frame)

            key = cv2.waitKey(0) & 0xFF
            if key == ord("s"):
                videoFrames = []
                startIndex = frameIndex
                print("Starting at frame {0}".format(startIndex))
                continue
            elif key == ord("e"):
                if frameIndex < startIndex:
                    print("End index selected before start index. Please select a start index [s] first.")
                    continue

                print("Ending at frame {0}".format(frameIndex))
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
                for thread in saveThreads:
                    thread.join()
                sys.exit(0)
        
        # Prompt the user for the class to store the segment as
        class_name = easygui.enterbox("What is the class of this data? (zero, eight, etc)")
        segment_dir = dest + '/segments/' + class_name
        if not os.path.exists(segment_dir):
            os.makedirs(segment_dir)

        while len(saveThreads) > 0:
            if not saveThreads[0].is_alive():
                doneThread = saveThreads.popleft()
                doneThread.join()
            else:
                break

        print("Loading and saving {0} frame segment...".format(frameIndex + 1 - startIndex))
        segment_filename = segment_dir + '/' + str(clipIndex)
        save_thread = threading.Thread(target=save, args=(video, startIndex, frameIndex + 1, segment_filename))
        save_thread.start()
        saveThreads.append(save_thread)

        if trace:
            path_dir = dest + '/paths/' + class_name
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            path_filename = path_dir + '/' + str(clipIndex)

            path_save_thread = saveTracePathFromVideoSource(video,
                initial_frame=startIndex, frames_count=(frameIndex + 1) - startIndex,
                save_dest=path_filename, height=height, fps=fps)
            path_save_thread.start()
            saveThreads.append(path_save_thread)

        clipIndex += 1

if __name__ == '__main__':
    segment()