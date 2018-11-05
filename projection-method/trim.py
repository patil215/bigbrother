import os
import pickle
import random
import sys
import threading
from collections import deque

import click
import cv2
import easygui
import imutils

from fileutils import read_video_frames, read_obj, write_video_frames, write_obj
from motiontrack import Tracker
from readvideo import save_tracepath_from_raw_video, tracepath_from_frames
from tracepoint import TracePath, TracePoint
from vizutils import draw_tracepoints, request_bounding_box


def make_process_segment_thread(video_file, dest_path, start_index, end_index):
    return threading.Thread(
        target=write_video_frames,
        args=(video_file, start_index, end_index, dest_path)
    )

def make_process_path_thread(video_file, dest_path, start_index, end_index, viewport, height=700, fps=60):
    source = cv2.VideoCapture(video_file)
    source.set(1, start_index)
    ok, raw_frame = source.read()
    tracker = Tracker(raw_frame, 'CSRT', height)

    return threading.Thread(
        target=save_tracepath_from_raw_video,
        args=(
            source,
            start_index,
            end_index,
            tracker,
            dest_path,
            viewport,
            fps
        )
    )

def generate_random_bounding_boxes(seed_box, count, width, height):
    # (x, y, width in x, height in y)
    bboxes = []
    for i in range(0, count):
        RAND_RANGE = 3
        original_bbox = seed_box

        new_x = min(original_bbox[0] + random.randint(-RAND_RANGE, RAND_RANGE), width)
        new_y = min(original_bbox[1] + random.randint(-RAND_RANGE, RAND_RANGE), height)
        new_width = original_bbox[2] + random.randint(-RAND_RANGE, RAND_RANGE)
        if new_x + new_width > width:
            new_width = width - new_x
        new_height = original_bbox[3] + random.randint(-RAND_RANGE, RAND_RANGE)
        if new_y + new_height > height:
            new_height = height - new_y

        bboxes.append((new_x, new_y, new_width, new_height))
    return bboxes

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
@click.option("-f", "--fps", default=60.0, required=False, help="Framerate of the input video, used for trace paths")
@click.option("-s", "--start", default=0, required=False, help="Start at this index")
@click.option("-v", "--vertical", default=None, required=False,
    help="""Vertical video to correlate.
        It is assumed the length of the vertical video matches the original video.
        It is also assumed that the framerate of the vertical video is the same as the original.
        """)
@click.option("-o", "--offset", type=click.INT, required=False, default=None, help="How many frames behind vertical is compared to horizontal")
@click.option("-z", "--viewport_horizontal", nargs=2, default=(20, 35), help="Size of viewable viewport for top-down video (X, then Y) in cm")
@click.option("-x", "--viewport_vertical", nargs=2, default=(34, 19), help="Size of viewport for vertical video (Y, then Z) in cm. Right now, only Z is used.")
def segment(video, height, dest, trace, debug, fps, start, vertical, offset, viewport_horizontal, viewport_vertical):
    if not os.path.exists(video):
        print("Invalid video to trim provided!")
        sys.exit(1)
    if vertical and not os.path.exists(vertical):
        print("Invalid vertical video provided!")
        sys.exit(1)
    if vertical and not offset:
        print("Please supply an offset!")
        sys.exit(1)

    viewport_horizontal = viewport_horizontal if vertical else None
    viewport_vertical = viewport_vertical if vertical else None

    VIDEO_SOURCE = cv2.VideoCapture(video)
    VIDEO_VERTICAL = cv2.VideoCapture(vertical) if vertical else None

    frame_index = 0
    start_index = 0
    clip_index = start
    threads = deque()
    tracepath_files_to_merge = []  # Array of tuples

    while True:
        while True:
            VIDEO_SOURCE.set(1, frame_index)
            ok, raw_frame = VIDEO_SOURCE.read()
            if not ok:
                safe_quit(threads, tracepath_files_to_merge, 0)

            frame = imutils.resize(raw_frame, height=height)
            cv2.imshow("Clipper", frame)

            key = cv2.waitKey(0) & 0xFF
            if key == ord("s"):
                videoFrames = []
                start_index = frame_index
                print("[{0} - ?] Starting at frame {0}".format(start_index))
                continue
            elif key == ord("e"):
                if frame_index < start_index:
                    print("End index selected before start index. Please select a start index [s] first.")
                    continue

                print("[{0} - {1}] Ending at frame {1}".format(start_index, frame_index))
                break
            elif key == ord("b"):
                frame_index = frame_index - 1 if frame_index > 0 else frame_index
                continue
            elif key == ord("n"):
                frame_index += 1
                continue
            elif key == ord("f"):
                frame_index += 20
                continue
            elif key == ord("v"):
                frame_index += 400
                continue
            elif key == ord("z"):
                frame_index = frame_index - 400 if frame_index > 400 else frame_index
                continue
            elif key == ord("a"):
                frame_index = frame_index - 20 if frame_index > 20 else frame_index
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
            .format(start_index, frame_index, frame_index + 1 - start_index)
        )
        segment_save_dest = "{}/segments/{}/{}.segment".format(dest, class_name, str(clip_index))
        segment_save_thread = make_process_segment_thread(
            video,
            segment_save_dest,
            start_index,
            frame_index
        )
        segment_save_thread.start()
        threads.append(segment_save_thread)

        # Save the paths
        # Get the TracePath for the video segment
        path_save_video_dest = "{}/paths/{}/{}.path".format(dest, class_name, str(clip_index))

        if trace:
            # When debugging, the bounding box can be redrawn until satisfied
            # with the resulting trace. Confirm with Enter.
            if debug:
                print("DEBUG trace path mode enabled. Loading segment...")
                target_segment = read_video_frames(video, start_index, frame_index)
                initial_frame = target_segment[0]
                proposed_paths = []

                print("Generating bounding boxes...")
                seed_bbox = request_bounding_box(target_segment[0], height)
                bboxes = [seed_bbox] + generate_random_bounding_boxes(seed_bbox, 4, initial_frame.shape[1], initial_frame.shape[0])

                print("Performing motion tracking...")
                for bbox in bboxes:
                    proposed_paths.append(tracepath_from_frames(target_segment, viewport_horizontal, height, fps, 
                        tracker=Tracker(target_segment[0], 'CSRT', height, bbox=bbox)))

                path_index = 0
                for path_obj in proposed_paths:
                    draw_tracepoints(path_obj, title="Proposed Path")

                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        safe_quit(threads, tracepath_files_to_merge, 0)
                    elif key == ord('\r'):
                        write_obj(path_save_video_dest.replace(".path", "-{}.path".format(path_index)), path_obj)
                        path_index += 1
                    # for any other key, skip
                cv2.destroyWindow("Proposed Path")

                print("Saved {} paths to {}".format(path_index, path_save_video_dest))

            # If not debugging, trace and save asynchronously
            else:
                path_save_video_thread = make_process_path_thread(
                    video,
                    path_save_video_dest,
                    start_index,
                    frame_index,
                    viewport_horizontal,
                    fps=fps
                )
                path_save_video_thread.start()
                threads.append(path_save_video_thread)

        if trace and VIDEO_VERTICAL:
            # We need to find the TracePath for both segments, and schedule a merge later.
            path_save_vertical_dest = "{}/paths_vertical/{}/{}.vsegment".format(dest, class_name, str(clip_index))
            path_save_vertical_thread = make_process_path_thread(
                vertical,
                path_save_vertical_dest,
                start_index + offset,
                frame_index + offset,
                viewport_vertical,
                fps=fps
            )
            path_save_vertical_thread.start()
            threads.append(path_save_vertical_thread)

            tracepath_files_to_merge.append((path_save_video_dest, path_save_vertical_dest))

        clip_index += 1

if __name__ == '__main__':
    segment()
