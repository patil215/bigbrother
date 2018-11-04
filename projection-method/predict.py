import click
import imutils
from fileutils import read_training_data, read_obj
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
from classify import classifyDTW, computeSegment, prepData
from vizutils import draw_tracepoints, plotPath
from readvideo import getTracePathFromVideoFile


def playVideo(filename, height, fps):
	video_segment = read_obj(filename)
	print("Video is " + str(len(video_segment) * (1 / 29.97)) + " seconds long") # TODO hardcoded FPS
	for frame_index in range(len(video_segment)):
		frame = video_segment[frame_index]
		frame = imutils.resize(frame, height)
		cv2.imshow("video", frame)
		cv2.waitKey(0)

@click.command()
@click.argument('filename')
@click.option('-h', '--height', help="Video display height", default=700)
@click.option('-f', '--fps', help="Input video framerate", default=29.97)
@click.option('--preview', help="Just view the video", is_flag=True)
@click.option('-d', '--data', help="Location of the data directory", default="data")
@click.option('-a', '--angle', help="Camera position in degrees", nargs=3, default=(0, 0, 0))
def predict(filename, height, fps, data, angle, preview):
	if not os.path.exists(filename):
		print("Invalid filename provided!")
		return
	if preview:
		playVideo(filename, height, fps)
		return
	if not os.path.exists(data):
		print("Invalid data directory provided!")
		return

	data = read_training_data(data)

	x, y, z = [math.radians(int(d)) for d in angle]
	transform = eulerAnglesToRotationMatrix(np.array([x, y, z]))
	prepData(data, transform)

	video_data = getTracePathFromVideoFile(filename, height=height, fps=fps)
	#video_data.transform(transform)
	video_data.normalize()

	print(classifyDTW(data, video_data))
	#print(computeSegment(video_data, data, 3))

if __name__ == "__main__":
	predict()