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

from fileutils import (get_next_file_number, make_dir, read_obj,
                       read_video_frames, write_obj, write_video_frames)
from motion_track import Tracker
from read_video import save_tracepath_from_raw_video, tracepath_from_frames
from tracepoint import TracePath, TracePoint
from vizutils import (draw_tracepoints, generate_random_bounding_boxes,
                      request_bounding_box)


def spawn_path_track_thread(video_file, dest_full_path, start_index, end_index, viewport, fps, height, checkpoints=None):
    checkpoints = checkpoints if checkpoints is not None else set()

    source = cv2.VideoCapture(video_file)
    source.set(1, start_index)
    ok, raw_frame = source.read()
    tracker = Tracker(raw_frame, height)

    return threading.Thread(
        target=save_tracepath_from_raw_video,
        args=(
            source,
            dest_full_path,
            start_index,
            end_index,
            tracker,
            viewport,
            fps,
            checkpoints,
        )
    )

def merge_tracepaths(to_merge):
    for path_save_video_dest, path_save_vertical_dest in to_merge:
        tracepath_xy = read_obj(path_save_video_dest)
        tracepath_z = read_obj(path_save_vertical_dest)

        if tracepath_xy.checkpoint_indices != tracepath_z.checkpoint_indices:
            print("Warning: checkpoint indices do not match")

        # Merge the two tracepaths
        merged_tracepath = TracePath(checkpoint_indices=tracepath_xy.checkpoint_indices)
        if len(tracepath_xy.path) != len(tracepath_z.path):
            print("Warning: path lengths are not equal")

        for i in range(len(tracepath_xy.path)):
            xy = tracepath_xy.path[i]
            z = tracepath_z.path[i]
            is_checkpoint = i in tracepath_xy.checkpoint_indices
            merged_tracepath.add(TracePoint((xy.pos[0], xy.pos[1], z.pos[1]), xy.t), is_checkpoint)

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
    # The input array should be empty if the vertical flag is not specified.
    merge_tracepaths(tracepaths_to_merge)

    sys.exit(exit_code)

@click.command()
@click.argument("video")
@click.option("-c", "--compressed", default=None, required=False, help="Compressed video to display for fast loading")
@click.option("-h", "--height", default=700, required=False, help="Video preview display height")
@click.option("-d", "--dest", default="data", required=False, help="Destination folder for segments and paths")
@click.option("-v", "--vertical", default=None, required=False,
    help="""Vertical video to correlate.
        It is assumed the length of the vertical video matches the original video.
        It is also assumed that the framerate of the vertical video is the same as the original.
        """)
@click.option("-o", "--offset", type=click.INT, required=False, default=None, help="How many frames behind vertical is compared to horizontal")
@click.option("-z", "--viewport_horizontal", nargs=2, default=(20, 35), help="Size of viewable viewport for top-down video (X, then Y) in cm")
@click.option("-x", "--viewport_vertical", nargs=2, default=(34, 19), help="Size of viewport for vertical video (Y, then Z) in cm. Right now, only Z is used.")
def segment(video, compressed, height, dest, vertical, offset, viewport_horizontal, viewport_vertical):
    if not os.path.exists(video):
        print("Invalid video to trim provided!")
        sys.exit(1)
    if compressed and not os.path.exists(compressed):
        print("Invalid compressed video provided!")
        sys.exit(1)
    if vertical and not os.path.exists(vertical):
        print("Invalid vertical video provided!")
        sys.exit(1)
    if vertical and not offset:
        print("Please supply an offset!")
        sys.exit(1)

    viewport_horizontal = viewport_horizontal if vertical else None
    viewport_vertical = viewport_vertical if vertical else None

    DEST_BASE_DIR = dest

    # Load videos
    VIDEO_SOURCE = cv2.VideoCapture(video)
    VIDEO_COMPRESSED_SOURCE = cv2.VideoCapture(compressed) if compressed else None
    VIDEO_VERTICAL = cv2.VideoCapture(vertical) if vertical else None

    # sanity check FPSes
    fps = VIDEO_SOURCE.get(cv2.CAP_PROP_FPS)
    print("FPS of source: {}".format(fps))
    if VIDEO_COMPRESSED_SOURCE:
        fps_compressed = VIDEO_COMPRESSED_SOURCE.get(cv2.CAP_PROP_FPS)
        print("FPS of compressed: {}".format(fps_compressed))

    if compressed and not fps == fps_compressed:
        print("FPS of source and compressed videos don't match ({} vs {})".format(fps, fps_compressed))
        sys.exit(1)

    if VIDEO_VERTICAL:
        fps_vertical = VIDEO_VERTICAL.get(cv2.CAP_PROP_FPS)
        print("FPS of vertical: {}".format(fps_vertical))

    if vertical and not fps == fps_vertical:
        print("FPS of both videos don't match ({} vs {})".format(fps, fps_vertical))
        sys.exit(1)

    # display the compressed video instead of the raw video if there is one
    if VIDEO_COMPRESSED_SOURCE:
        VIDEO_SOURCE = VIDEO_COMPRESSED_SOURCE

    frame_index = 0
    start_index = 0
    checkpoint_indices = set()

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
                start_index = frame_index
                checkpoint_indices = set()

                print("[{0} - ?] Starting at frame {0}".format(start_index))
                print("Skipping ahead 5 frames...")
                frame_index += 5
            elif key == ord("e"):
                if frame_index < start_index:
                    print("End index selected before start index. Please select a start index [s] first.")
                    continue
                print("[{0} - {1}] Ending at frame {1}".format(start_index, frame_index))
                break
            elif key == ord('c'):
                # mark checkpoint
                if frame_index not in checkpoint_indices and frame_index > start_index:
                    checkpoint_indices.add(frame_index)
                    print("[{} - ?] Marking checkpoint #{} at frame {} (+{})".format(
                        start_index, len(checkpoint_indices), frame_index, frame_index - start_index
                    ))
                    print("Skipping ahead 5 frames...")
                    frame_index += 5
            elif key == ord("b"):
                frame_index = frame_index - 1 if frame_index > 0 else frame_index
            elif key == ord("n"):
                frame_index += 1
            elif key == ord("h"):
                frame_index = frame_index - 2 if frame_index > 1 else frame_index
            elif key == ord("j"):
                frame_index += 2
            elif key == ord("f"):
                frame_index += 20
            elif key == ord("v"):
                frame_index += 400
            elif key == ord("z"):
                frame_index = frame_index - 400 if frame_index > 400 else frame_index
            elif key == ord("a"):
                frame_index = frame_index - 20 if frame_index > 20 else frame_index
            elif key == ord("q"):
                safe_quit(threads, tracepath_files_to_merge, 0)

        # Prompt the user for the class to store the segment as
        num_checkpoints = len(checkpoint_indices)
        if num_checkpoints == 0:
            print("INFO: zero checkpoints specified, assuming unpadded single digit")
        elif num_checkpoints % 2 == 1:
            print("WARNING: odd number of checkpoints ({}). Assuming end frame is final checkpoint...".format(num_checkpoints))
            checkpoint_indices.add(frame_index)

        num_digits = int(len(checkpoint_indices) / 2) if len(checkpoint_indices) > 0 else 1
        concat_class_name = ""
        class_names = []
        for i in range(3):
            concat_class_name = easygui.enterbox("Please specify {} digits formatted like 'zero_one_two' etc.".format(num_digits))
            class_names = concat_class_name.split('_')

            if len(class_names) == num_digits:
                break
            print("WARNING: expected {} digits but specified {} (input string: {})".format(num_digits, len(class_names), concat_class_name))

        # Clean up any finished threads
        while len(threads) > 0 and not threads[0].is_alive():
            threads.popleft().join()

        # Generate new file destinations.
        # The file number is the `max` of all the next file numbers in possible destinations, for consistency.
        digit_qty_folder = "{}/{}_digits/".format(DEST_BASE_DIR, len(class_names))
        class_save_folder = digit_qty_folder + concat_class_name
        path_save_folder = "{}/paths".format(class_save_folder)
        vertical_path_save_folder = "{}/paths_vertical".format(class_save_folder)

        file_number = max(
            get_next_file_number(path_save_folder),
            get_next_file_number(vertical_path_save_folder)
        )

        path_save_path = "{}/{}.path".format(path_save_folder, file_number)
        vertical_path_save_path = "{}/{}.vpath".format(path_save_folder, file_number)

        print(
            "[{0} - {1}] Creating path from segment of length {2}..."
            .format(start_index, frame_index, frame_index + 1 - start_index)
        )

        path_save_video_thread = spawn_path_track_thread(
            video,
            path_save_path,
            start_index,
            frame_index,
            viewport_horizontal,
            fps,
            height,
            checkpoints=checkpoint_indices
        )
        path_save_video_thread.start()
        threads.append(path_save_video_thread)

        if vertical:
            # We need to find the TracePath for both segments, and schedule a merge later.
            path_save_vertical_thread = spawn_path_track_thread(
                vertical,
                vertical_path_save_path,
                start_index + offset,
                frame_index + offset,
                viewport_vertical,
                fps,
                height
                checkpoints=set([i + offset for i in checkpoint_indices])
            )
            path_save_vertical_thread.start()
            threads.append(path_save_vertical_thread)

            tracepath_files_to_merge.append((path_save_path, vertical_path_save_path))

if __name__ == '__main__':
    segment()
