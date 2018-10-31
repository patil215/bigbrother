import cv2
import imutils
import pickle
import sys
from fileutils import read_obj, write_obj, save_video_clip, load_video_clip
from motiontrack import Tracker
import click
from readvideo import asyncTrackSave, getTracePathFromFrames
import os
import easygui
import sys
import threading
from collections import deque
from vizutils import draw_tracepoints
from tracepoint import TracePath, TracePoint

def make_process_segment_thread(video_file, dest_path, start_index, end_index):
    return threading.Thread(
        target=save_video_clip,
        args=(video_file, start_index, end_index, dest_path)
    )

def make_process_path_thread(video_file, dest_path, start_index, end_index, height=700, fps=60):
    source = cv2.VideoCapture(video_file)
    source.set(1, start_index)
    ok, raw_frame = source.read()
    tracker = Tracker(raw_frame, 'CSRT', height)

    return threading.Thread(
        target=asyncTrackSave,
        args=(
            source,
            start_index,
            end_index,
            tracker,
            dest_path,
            fps
        )
    )

def merge_tracepaths(to_merge):
    for path_save_video_dest, path_save_vertical_dest in to_merge:
        tracepath_xy = read_obj(path_save_video_dest)
        tracepath_z = read_obj(path_save_vertical_dest)

        # Merge the two tracepaths
        merged_tracepath = TracePath()
        for xy, z in zip(tracepath_xy.path, tracepath_z.path):
            # The Y position in Z is the height, hence z.pos[1]
            merged_tracepath.add(TracePoint((xy.pos[0], xy.pos[1], z.pos[1]), xy.t))

        # Write the final result
        path_save_merged_dest = path_save_video_dest.replace("paths", "paths_merged")
        write_obj(path_save_merged_dest, merged_tracepath)

        print("Merged tracepaths at {} and {} ".format(path_save_video_dest, path_save_vertical_dest))

def safe_quit(threads, tracepaths_to_merge, exit_code):
    print("Joining threads in order to quit...")
    for thread in threads:
        thread.join()

    print("Merging tracepath...")
    # Now, we need to merge any TracePaths if we have the VERTICAL flag passed in.
    merge_tracepaths(tracepaths_to_merge)

    sys.exit(0)

@click.command()
@click.argument("video")
@click.option("-h", "--height", default=700, required=False, help="Video preview display height")
@click.option("-d", "--dest", default="data", required=False, help="Destination folder for segments and paths")
@click.option("-t", "--trace", is_flag=True, help="Enable trace path recording mode")
@click.option("-d", "--debug", is_flag=True, help="Enable trace path debugging", default=False, required=False)
@click.option("-f", "--fps", default=60, required=False, help="Framerate of the input video, used for trace paths")
@click.option("-s", "--start", default=0, required=False, help="Start at this index")
@click.option("-v", "--vertical", default=None, required=False,
    help="""Vertical video to correlate.
        It is assumed the length of the vertical video matches the original video.
        It is also assumed that the framerate of the vertical video is the same as the original.
        """)
def segment(video, height, dest, trace, debug, fps, start, vertical):
    """ TODO FOR PROPER CORRELATION:
    - Make aspect ratio normalization work with Z
    - Account for different scales with the XY and Z
    - Make the merging not rely on a join
    - Account for different framerates, and not rely on clipping the two clips exactly correctly
    """

    if not os.path.exists(video):
        print("Invalid video to trim provided")
        sys.exit(1)
    if vertical and not os.path.exists(vertical):
        print("Invalid vertical video provided")
        sys.exit(1)

    VIDEO_SOURCE = cv2.VideoCapture(video)
    VIDEO_VERTICAL = cv2.VideoCapture(vertical) if vertical else None

    if VIDEO_VERTICAL and int(VIDEO_SOURCE.get(cv2.CAP_PROP_FRAME_COUNT)) != int(VIDEO_VERTICAL.get(cv2.CAP_PROP_FRAME_COUNT)):
        print("Vertical video length does not match original video!")
        sys.exit(1)

    frameIndex = 0
    clipIndex = start
    threads = deque()
    tracepath_files_to_merge = []  # Array of tuples

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
                print("[{0} - ?] Starting at frame {0}".format(startIndex))
                continue
            elif key == ord("e"):
                if frameIndex < startIndex:
                    print("End index selected before start index. Please select a start index [s] first.")
                    continue

                print("[{0} - {1}] Ending at frame {1}".format(startIndex, frameIndex))
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
                safe_quit(threads, tracepath_files_to_merge, 0)

        # Prompt the user for the class to store the segment as
        class_name = easygui.enterbox("What is the class of this data? (zero, eight, etc)")

        # Clean up any finished threads
        while len(threads) > 0 and not threads[0].is_alive():
            threads.popleft().join()

        # Save the segment from video
        print(
            "[{0} - {1}] Loading and saving {2} frame segment..."
            .format(startIndex, frameIndex, frameIndex + 1 - startIndex)
        )
        segment_save_dest = dest + '/segments/' + class_name + '/' + str(clipIndex)
        segment_save_thread = make_process_segment_thread(
            video,
            segment_save_dest,
            startIndex,
            frameIndex
        )
        segment_save_thread.start()
        threads.append(segment_save_thread)

        # Save the paths
        if trace:
            # Get the TracePath for the video segment
            path_save_video_dest = dest + '/paths/' + class_name + '/' + str(clipIndex)

            # When debugging, the bounding box can be redrawn until satisfied
            # with the resulting trace. Confirm with Enter.
            if debug:
                print("DEBUG trace path mode enabled. Loading segment...")
                target_segment = load_video_clip(video, startIndex, frameIndex)
                proposed_path = None

                while True:
                    proposed_path = getTracePathFromFrames(target_segment, height, fps)
                    draw_tracepoints(proposed_path, frame="Proposed Path")

                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        safe_quit(threads, tracepath_files_to_merge, 0)
                    if key == ord('\r'):
                        cv2.destroyWindow("Proposed Path")
                        cv2.waitKey(1)
                        break

                write_obj(path_save_video_dest, proposed_path)

            # If not debugging, trace and save asynchronously
            else:
                path_save_video_thread = make_process_path_thread(
                    video,
                    path_save_video_dest,
                    startIndex,
                    frameIndex,
                    fps=fps
                )
                path_save_video_thread.start()
                threads.append(path_save_video_thread)

        if trace and VIDEO_VERTICAL:
            # We need to find the TracePath for both segments, and create a TracePath with both.
            # Unfortunately, this is difficult to multithread.
            path_save_vertical_dest = dest + '/paths_vertical/' + class_name + '/' + str(clipIndex)
            path_save_vertical_thread = make_process_path_thread(
                vertical,
                path_save_vertical_dest,
                startIndex,
                frameIndex,
            )
            path_save_vertical_thread.start()
            threads.append(path_save_vertical_thread)

            tracepath_files_to_merge.append((path_save_video_dest, path_save_vertical_dest))

        clipIndex += 1

if __name__ == '__main__':
    segment()