import click
import imutils
from fileutils import readData, read_obj, readVideosLazy, write_obj
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
from readvideo import getTracePathFromFrames


@click.command()
@click.argument('test_dir')
@click.option('-h', '--height', help="Video display height", default=700)
@click.option('-f', '--fps', help="Input video framerate", default=29.97)
@click.option('-d', '--data', help="Location of the data directory", default="data")
@click.option('-a', '--angle', help="Camera position in degrees", nargs=3, default=(0, 0, 0))
def predict(test_dir, height, fps, data, angle):
	if not os.path.exists(test_dir):
		print("Invalid test directory provided!")
		return
	if not os.path.exists(data):
		print("Invalid data directory provided!")
		return

	videos = readVideosLazy(test_dir)
	data = readData(data)

	x, y, z = [math.radians(int(d)) for d in angle]
	transform = eulerAnglesToRotationMatrix(np.array([x, y, z]))
	prepData(data, transform)

	correct = 0
	correct_top3 = 0
	correct_top5 = 0
	total = 0
	# Go through videos and classify / score them!!!1
	for video_class in videos:
		class_videos = videos[video_class]
		for video_path in class_videos:
			video = read_obj(video_path)
			video_base_name = video_path.split("/")[-1]

			metadata_path = test_dir + '/' + video_class + '/.' + video_base_name + '.meta'
			bbox = read_obj(metadata_path)
			if not bbox:
				# Prompt for bounding box and store as metadata
				tracker = Tracker(video[0], height=height)
				write_obj(metadata_path, tracker.bbox)
			else:
				tracker = Tracker(video[0], height=height, bbox=bbox)

			video_data = getTracePathFromFrames(video, height=height, fps=fps, tracker=tracker)
			video_data.normalize()

			actual = classifyDTW(data, video_data)
			top3 = [thing[0] for thing in actual[:3]]
			top5 = [thing[0] for thing in actual[:5]]
			print("Expected: {} Actual: {}".format(video_class, actual[0][0]))
			if actual[0][0] == video_class:
				correct += 1
			if video_class in top3:
				correct_top3 += 1
			if video_class in top5:
				correct_top5 += 1
			total += 1

	print("Correct: {} Correct (top 3): {} Correct (top 5): {} Total: {}".format(correct, correct_top3, correct_top5, total))


if __name__ == "__main__":
	predict()