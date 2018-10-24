from fileutils import readData, read_obj
from tracepoint import TracePath, TracePoint
import imutils
from motiontrack import Tracker

def getTracePathFromFrames(video_segment, height=700, fps=60):
	tracepath = TracePath()
	initial_frame = video_segment[0]
	tracker = Tracker(initial_frame, 'CSRT', height)

	for frame_index in range(len(video_segment)):
		frame = video_segment[frame_index]

		bbox = tracker.track(frame)

		timestamp = (1000.0 / fps) * frame_index

		# Append center of bounding box
		x = bbox[0] + (bbox[2] / 2) # TODO this might actually be Y, check it
		y = bbox[1] + (bbox[3] / 2)
		tracepath.add(TracePoint((x, y, 0), timestamp))

	return tracepath


def getTracePathFromVideoFile(filename):
	tracepath = TracePath()
	video_segment = read_obj(filename)
	return getTracePathFromFrames(video_segment)
