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
from classify import classifyDTW
from vizutils import draw_tracepoints, plotPath
from readvideo import getTracePathFromVideoFile

def prepData(data, R):
	for category in data:
		for tracepath in data[category]:
			tracepath.transform(R)
			tracepath.normalize()

def playVideo(filename):
	video_segment = read_obj(filename)
	for frame_index in range(len(video_segment)):
		frame = video_segment[frame_index]
		frame = imutils.resize(frame, width=1600)
		cv2.imshow("video", frame)
		cv2.waitKey(0)

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
@click.option('--preview', help="Just view the video", is_flag=True)
@click.option('-d', '--data', help="Location of the data directory", default="data")
@click.option('-a', '--angle', help="Camera position in degrees", nargs=3, default=(0, 0, 0))
def predict(filename, data, angle, preview):
	if not os.path.exists(filename):
		print("Invalid filename provided!")
	if not os.path.exists(data):
		print("Invalid data directory provided!")

	if preview:
		playVideo(filename)
		return

	x, y, z = [math.radians(int(d)) for d in angle]
	transform = eulerAnglesToRotationMatrix(np.array([x, y, z]))
	print(transform)


	data = readData(data)
	prepData(data, transform)
	drawTransformed(data["zero"][0])
	drawTransformed(data["zero"][1])
	drawTransformed(data["one"][0])
	drawTransformed(data["one"][1])

	video_data = getTracePathFromVideoFile(filename)
	video_data.normalize()
	drawTransformed(video_data)

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

if __name__ == "__main__":
	predict()