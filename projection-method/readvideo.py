from fileutils import readData, read_obj
from tracepoint import TracePath, TracePoint
import imutils
from motiontrack import Tracker

def getTracePathFromFrames(video_segment):
	tracepath = TracePath()
	initial_frame = video_segment[0]
	initial_frame = imutils.resize(initial_frame, width=1600)
	tracker = Tracker(initial_frame, 'CSRT')

	for frame_index in range(len(video_segment)):
		frame = video_segment[frame_index]
		frame = imutils.resize(frame, width=1600)

		bbox = tracker.track(frame)

		timestamp = (1000.0 / 10) * frame_index # TODO don't hardcode
		
		"""p1 = (int(bbox[0]), int(bbox[1]))
		p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
		cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
		cv2.imshow("badd", frame)
		cv2.waitKey(0)"""


		# Append center of bounding box
		x = bbox[0] + (bbox[2] / 2) # TODO this might actually be Y, check it
		y = bbox[1] + (bbox[3] / 2)
		tracepath.add(TracePoint((x, y, 0), timestamp))

	return tracepath


def getTracePathFromVideoFile(filename):
	tracepath = TracePath()
	video_segment = read_obj(filename)
	return getTracePathFromFrames(video_segment)
