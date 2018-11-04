import math
import os
import time
from time import sleep

import click
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

from classify import classifyDTW, computeSegment, prepData
from fileutils import read_obj, write_obj, read_training_data, get_test_segment_tree, get_test_path_tree
from motiontrack import Tracker
from project import eulerAnglesToRotationMatrix
from readvideo import getTracePathFromFrames
from tracepoint import TracePath, TracePoint
from vizutils import draw_tracepoints, plotPath


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

	path_tree = get_test_path_tree(test_dir)
	video_tree = get_test_segment_tree(test_dir)
	training_data = read_training_data(data)

	x, y, z = [math.radians(int(d)) for d in angle]
	transform = eulerAnglesToRotationMatrix(np.array([x, y, z]))
	prepData(training_data, transform)

	correct = 0
	correct_top3 = 0
	correct_top5 = 0
	total = 0

	# Classify test paths.
	print("Classifying test paths...")
	for video_class in path_tree:
		class_paths = path_tree[video_class]
		for path_name in class_paths:
			path = read_obj("{}/{}/{}".format(test_dir, video_class, path_name))

			classifications = classifyDTW(training_data, path)
			top3 = [thing[0] for thing in classifications[:3]]
			top5 = [thing[0] for thing in classifications[:5]]
			print("Predicted: {}, Expected: {}".format(classifications[0][0], video_class))
			if classifications[0][0] == video_class:
				correct += 1
			if video_class in top3:
				correct_top3 += 1
			if video_class in top5:
				correct_top5 += 1
			total += 1

	# Record new paths and classify them for video clips without existing path files.
	print("Generating paths for new files...")
	for video_class in video_tree:
		class_videos = video_tree[video_class]
		for video_name in class_videos:
			# Ignore videos which already have a path associated
			if ".{}.path".format(video_name) in path_tree[video_class]:
				continue

			video = read_obj("{}/{}/{}".format(test_dir, video_class, video_name))
			metadata_path = "{}/{}/{}".format(test_dir, video_class, ".{}.meta".format(video_name))
			bbox = read_obj(metadata_path)

			if not bbox:
				# Prompt for bounding box and store as metadata
				tracker = Tracker(video[0], height=height)
				write_obj(metadata_path, tracker.bbox)
			else:
				tracker = Tracker(video[0], height=height, bbox=bbox)

			video_data = getTracePathFromFrames(video, height=height, fps=fps, tracker=tracker)
			video_data.normalize()
			write_obj("{}/{}/{}".format(test_dir, video_class, ".{}.path".format(video_name)), video_data)

			classifications = classifyDTW(training_data, video_data)
			top3 = [thing[0] for thing in classifications[:3]]
			top5 = [thing[0] for thing in classifications[:5]]
			print("Predicted: {}, Expected: {}".format(classifications[0][0], video_class))
			if classifications[0][0] == video_class:
				correct += 1
			if video_class in top3:
				correct_top3 += 1
			if video_class in top5:
				correct_top5 += 1
			total += 1

	print("Correct: {} Correct (top 3): {} Correct (top 5): {} Total: {}".format(correct, correct_top3, correct_top5, total))

if __name__ == "__main__":
	predict()
