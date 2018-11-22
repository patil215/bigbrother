import threading

import cv2
import imutils

from fileutils import read_obj, write_obj
from motiontrack import Tracker
from tracepoint import TracePath, TracePoint

def tracepoint_from_frame(frame, tracker, frame_index, fps, viewport=None):
	bbox = tracker.track(frame)

	timestamp = (1000.0 / fps) * frame_index

	x_pixels = bbox[0] + (bbox[2] / 2)
	y_pixels = bbox[1] + (bbox[3] / 2)

	if viewport:
		x_cm = (x_pixels / frame.shape[1]) * viewport[0]
		y_cm = (y_pixels / frame.shape[0]) * viewport[1]
		return TracePoint((x_cm, y_cm, 0), timestamp)
	return TracePoint((x_pixels, y_pixels, 0), timestamp)

def tracepath_from_frames(video_segment, fps, viewport=None, height=700, tracker=None):
	tracepath = TracePath()
	initial_frame = video_segment[0]
	tracker = tracker if tracker else Tracker(initial_frame, height)

	for frame_index in range(len(video_segment)):
		frame = video_segment[frame_index]
		tracepath.add(tracepoint_from_frame(frame, tracker, frame_index, fps, viewport))

	return tracepath

def save_tracepath_from_raw_video(source, dest_path, start_index, end_index, tracker, viewport, fps, checkpoints):
	print("[{0} - {1}] Beginning tracking...".format(start_index, end_index))
	tracepath = TracePath()
	for i in range(end_index - start_index + 1):
		current_frame_index = start_index + i
		source.set(1, current_frame_index)
		ok, raw_frame = source.read()

		tracepath.add(tracepoint_from_frame(raw_frame, tracker, i, fps, viewport),
			checkpoint=current_frame_index in checkpoints
		)

	print("[{0} - {1}] {2} length path tracking complete, saving to {3}".format(start_index, end_index, len(tracepath.path), dest_path))
	write_obj(dest_path, tracepath)
	print("[{0} - {1}] Path saved successfully to {2}".format(start_index, end_index, dest_path))
