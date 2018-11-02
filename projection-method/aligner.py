import cv2
import click
import os
import imutils
import sys
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np

def dec(val, amount):
    return val - amount if val > amount else val

def timeFromFrame(frame, fps):
    return frame / fps


"""
Lets you easily align two video segments - a horizontal segment, and a vertical segment - for tracing.
"""
@click.command()
@click.argument("video_horizontal")
@click.argument("video_vertical")
@click.argument("video_horizontal_dest")
@click.argument("video_vertical_dest")
@click.option("-h", "--height", default=500, type=click.INT, required=False)
def align(video_horizontal, video_vertical, video_horizontal_dest, video_vertical_dest, height):
    if not os.path.exists(video_horizontal) or not os.path.exists(video_vertical):
        print("Invalid paths provided for videos!")
        sys.exit(1)


    video_a = cv2.VideoCapture(video_horizontal)
    video_b = cv2.VideoCapture(video_vertical)

    video_a_length = int(video_a.get(cv2.CAP_PROP_FRAME_COUNT))
    video_b_length = int(video_b.get(cv2.CAP_PROP_FRAME_COUNT))
    print("first video has {} frames, second: {}".format(video_a_length, video_b_length))

    framerate_a = video_a.get(cv2.CAP_PROP_FPS)
    framerate_b = video_b.get(cv2.CAP_PROP_FPS)
    if framerate_a != framerate_b:
        print("ERROR: framerates differ ({} vs {})".format(framerate_a, framerate_b))
        sys.exit(1)

    start_a, start_b = (0, 0)
    end_a, end_b = (0, 0)
    ind_a = 0
    ind_b = 0
    while True:
        video_a.set(1, ind_a)
        video_b.set(1, ind_b)
        ok, frame_a = video_a.read()
        ok2, frame_b = video_b.read()
        if not ok or not ok2:
            print("BAD")
            continue

        
        frame_a = imutils.resize(frame_a, height=height)
        frame_b = imutils.resize(frame_b, height=height)

        print("")
        print("index_a: {} index_b: {}".format(ind_a, ind_b))
        print("start_a: {} end_a: {}, start_b: {}, end_b: {}".format(start_a, end_a, start_b, end_b))
        print("offset between the two indices: {}".format(index_b - index_a))
        print("{} vs {} frames".format((end_a - start_a), (end_b - start_b)))
        cv2.imshow("Trimmer", np.hstack((frame_a, frame_b)))

        key = cv2.waitKey(0) & 0xFF
        if key == ord("a"):
            ind_a = dec(ind_a, 20)
        elif key == ord("s"):
            ind_a = dec(ind_a, 1)
        elif key == ord("d"):
            ind_a += 1
        elif key == ord("f"):
            ind_a += 20
        elif key == ord("v"):
            ind_a += 1000
        elif key == ord("c"):
            ind_a = dec(ind_a, 1000)
        elif key == ord("m"):
            ind_b += 1000
        elif key == ord("n"):
            ind_b = dec(ind_b, 1000)
        elif key == ord("j"):
            ind_b = dec(ind_b, 20)
        elif key == ord("k"):
            ind_b = dec(ind_b, 1)
        elif key == ord("t"):
            ind_a = video_a_length - 20
        elif key == ord("y"):
            ind_b = video_b_length - 20
        elif key == ord("l"):
            ind_b += 1
        elif key == ord(";"):
            ind_b += 20
        elif key == ord("z"):
            start_a = ind_a
        elif key == ord("x"):
            end_a = ind_a
        elif key == ord(","):
            start_b = ind_b
        elif key == ord("."):
            end_b = ind_b
        elif key == ord("q"):
            if end_a - start_a != end_b - start_b:
                print("Segment lengths are not equal!")
                continue

            a_start_time = timeFromFrame(start_a, framerate_a)
            a_end_time = timeFromFrame(end_a, framerate_a)
            b_start_time = timeFromFrame(start_b, framerate_b)
            b_end_time = timeFromFrame(end_b, framerate_b)
            print("{} {} {} {}".format(a_start_time, a_end_time, b_start_time, b_end_time))
            # Save the properly aligned segments
            ffmpeg_extract_subclip(video_horizontal, a_start_time, a_end_time, targetname=video_horizontal_dest)
            ffmpeg_extract_subclip(video_vertical, b_start_time, b_end_time, targetname=video_vertical_dest)
            break

if __name__ == '__main__':
    align()