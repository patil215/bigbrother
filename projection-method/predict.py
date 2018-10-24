import click
import imutils
from fileutils import readData, read_obj
import matplotlib.pyplot as plt
import time
from time import sleep
import cv2
import os
import math
from project import eulerAnglesToRotationMatrix
import numpy as np
from motiontrack import Tracker
from tracepoint import TracePath, TracePoint
from classify import classifyDTW, computeSegment
from vizutils import draw_tracepoints, plotPath
from readvideo import getTracePathFromVideoFile

def prepData(data, R):
	for category in data:
		for tracepath in data[category]:
			tracepath.transform(R)
			tracepath.normalize()

def playVideo(filename):
	video_segment = read_obj(filename)
	print("Video is " + str(len(video_segment) * (1 / 29.97)) + " seconds long") # TODO hardcoded FPS
	for frame_index in range(len(video_segment)):
		frame = video_segment[frame_index]
		frame = imutils.resize(frame, height=700)
		cv2.imshow("video", frame)
		cv2.waitKey(0)

@click.command()
@click.argument('filename')
@click.option('--preview', help="Just view the video", is_flag=True)
@click.option('-d', '--data', help="Location of the data directory", default="data")
@click.option('-a', '--angle', help="Camera position in degrees", nargs=3, default=(0, 0, 0))
def predict(filename, data, angle, preview):
	if not os.path.exists(filename):
		print("Invalid filename provided!")
		return
	if not os.path.exists(data):
		print("Invalid data directory provided!")
		return
	if preview:
		playVideo(filename)
		return

	data = readData(data)

	#draw_tracepoints(data["zero"][0])
	#draw_tracepoints(data["zero"][1])
	#draw_tracepoints(data["one"][0])
	#draw_tracepoints(data["one"][1])

	x, y, z = [math.radians(int(d)) for d in angle]
	transform = eulerAnglesToRotationMatrix(np.array([x, y, z]))

	draw_tracepoints(data["zero"][0])
	draw_tracepoints(data["zero"][1])

	draw_tracepoints(data["one"][0])
	draw_tracepoints(data["one"][1])
	draw_tracepoints(data["seven"][0])
	draw_tracepoints(data["seven"][1])
	draw_tracepoints(data["eight"][0])
	draw_tracepoints(data["eight"][1])

	prepData(data, transform)
	#draw_tracepoints(data["zero"][1])
	#draw_tracepoints(data["one"][0])
	#draw_tracepoints(data["one"][1])

	video_data = getTracePathFromVideoFile(filename)
	video_data.normalize()
	#draw_tracepoints(video_data)

	"""plotPath(video_data, 0, 'red')
	plotPath(video_data, 1, 'blue')
	plotPath(data["zero"][0], 0, 'yellow')
	plotPath(data["zero"][0], 1, 'green')
	plotPath(data["one"][0], 0, 'black')
	plotPath(data["one"][0], 1, 'brown')
	plt.show()"""

	"""plotPath(path_zero, 1, 'r')
	plotPath(path_one, 1, 'g')
	plotPath(path_test, 1, 'b')"""

	print(classifyDTW(data, video_data))
	#print(computeSegment(video_data, data, 3))

if __name__ == "__main__":
	predict()