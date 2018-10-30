from fileutils import readData, read_obj, write_obj
from tracepoint import TracePath, TracePoint
import imutils
from motiontrack import Tracker
import threading
import cv2

def getTracePathFromFrames(video_segment, height=700, fps=60, tracker=None):
	tracepath = TracePath()
	initial_frame = video_segment[0]
	tracker = tracker if tracker else Tracker(initial_frame, 'CSRT', height) 

	for frame_index in range(len(video_segment)):
		frame = video_segment[frame_index]

		bbox = tracker.track(frame)

		timestamp = (1000.0 / fps) * frame_index

		# Append center of bounding box
		x = bbox[0] + (bbox[2] / 2) # TODO this might actually be Y, check it
		y = bbox[1] + (bbox[3] / 2)
		tracepath.add(TracePoint((x, y, 0), timestamp))

	return tracepath

def getTracePathFromVideoFile(filename, height=700, fps=60):
	video_segment = read_obj(filename)
	return getTracePathFromFrames(video_segment, height, fps)

def asyncTrackSave(source, initial_frame, end_frame, tracker, save_dest, fps=60):
	print("[{0} - {1}] Beginning tracking...".format(initial_frame, end_frame))
	tracepath = TracePath()
	for i in range(end_frame - initial_frame + 1):
		source.set(1, initial_frame + i)
		ok, rawFrame = source.read()

		bbox = tracker.track(rawFrame)

		timestamp = (1000.0 / fps) * i

		# Append center of bounding box
		x = bbox[0] + (bbox[2] / 2) # TODO this might actually be Y, check it
		y = bbox[1] + (bbox[3] / 2)
		tracepath.add(TracePoint((x, y, 0), timestamp))

	print("[{0} - {1}] {2} length path tracking complete, saving to {3}".format(initial_frame, end_frame, len(tracepath.path), save_dest))
	write_obj(save_dest, tracepath)
	print("[{0} - {1}] Path saved successfully to {2}".format(initial_frame, end_frame, save_dest))
