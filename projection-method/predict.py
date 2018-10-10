import click
import imutils
from fileutils import readData, read_obj
import matplotlib.pyplot as plt
import time
import cv2
import os
import math
from project import eulerAnglesToRotationMatrix
import numpy as np
from motiontrack import Tracker
from tracepoint import TracePath, TracePoint
from classify import classifyDTW
from vizutils import draw_tracepoints, plotPath

def prepData(data, R):
	for category in data:
		for tracepath in data[category]:
			tracepath.transform(R)
			tracepath.normalize()

def getTracePathFromVideo(filename):
	tracepath = TracePath()
	video_segment = read_obj(filename)
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

def create_blank(width, height, rgb_color=(0, 0, 0)):
	image = np.zeros((height, width, 3), np.uint8)
	image[:] = rgb_color
	return image

def drawTransformed(path):
	canvas = create_blank(512, 512, rgb_color=(0, 0, 0))

	draw_tracepoints(canvas, path, fit_canvas=True)

	cv2.imshow("canvas", canvas)
	cv2.waitKey()

@click.command()
@click.argument('filename')
@click.option('-d', '--data', help="Location of the data directory", default="data")
@click.option('-a', '--angle', help="Camera position in degrees", nargs=3, default=(0, 0, 0))
def predict(filename, data, angle):
	if not os.path.exists(filename):
		print("Invalid filename provided!")
	if not os.path.exists(data):
		print("Invalid data directory provided!")

	x, y, z = [math.radians(int(d)) for d in angle]
	transform = eulerAnglesToRotationMatrix(np.array([x, y, z]))
	print(transform)


	data = readData(data)
	prepData(data, transform)

	video_data = getTracePathFromVideo(filename)
	video_data.transform(transform)
	video_data.normalize()
	drawTransformed(video_data)

	plotPath(video_data, 0, 'r')
	plotPath(video_data, 1, 'blue')
	plotPath(data["zero"][0], 0, 'r')
	plotPath(data["zero"][0], 1, 'blue')
	#plotPath(data["one"][0], 0, 'b')
	#plotPath(data["one"][0], 1, 'b')
	#plotPath(data["eight"][0], 0, 'yellow')
	#plotPath(data["eight"][0], 1, 'yellow')
	plt.show()


	"""plotPath(path_zero, 1, 'r')
	plotPath(path_one, 1, 'g')
	plotPath(path_test, 1, 'b')"""


	print(classifyDTW(data, video_data))

if __name__ == "__main__":
	predict()