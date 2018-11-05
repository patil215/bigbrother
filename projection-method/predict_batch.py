import math
import os
import time
from time import sleep
from collections import defaultdict

import click
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from classify import classifyDTW, computeSegment, prep_data
from fileutils import read_obj, write_obj, read_training_data, get_test_segment_tree, get_test_path_tree
from motiontrack import Tracker
from project import eulerAnglesToRotationMatrix
from readvideo import tracepath_from_frames
from tracepoint import TracePath, TracePoint
from vizutils import draw_tracepoints, plotPath

def update_statistics(statistics, classifications, video_class):
	top3 = [thing[0] for thing in classifications[:3]]
	top5 = [thing[0] for thing in classifications[:5]]
	print("Predicted: {}, Expected: {}".format(classifications[0][0], video_class))
	if classifications[0][0] == video_class:
		statistics[video_class]["correct"] += 1
	if video_class in top3:
		statistics[video_class]["correct_top3"] += 1
	if video_class in top5:
		statistics[video_class]["correct_top5"] += 1
	statistics[video_class]["total"] += 1

def print_statistics(statistics):
	print("SUMMARY OF RESULTS:")
	print("Exact / Top 3 / Top 5 / Total: {} / {} / {} / {}".format(
			sum([statistics[c]["correct"] for c in statistics]),
			sum([statistics[c]["correct_top3"] for c in statistics]),
			sum([statistics[c]["correct_top5"] for c in statistics]),
			sum([statistics[c]["total"] for c in statistics]),
		)
	)
	print("BY CLASS:")
	print(
		tabulate(
			[[c, statistics[c]["correct"]/statistics[c]["total"], statistics[c]["correct_top3"], statistics[c]["correct_top5"], statistics[c]["total"]] for c in statistics],
			headers=["Class", "Percent correct", "Top 3", "Top 5", "Total"]
		)
	)

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
	prep_data(training_data, transform)

	statistics = defaultdict(lambda: defaultdict(int))

	# Classify test paths.
	print("Classifying test paths...")
	for video_class in path_tree:
		class_paths = path_tree[video_class]
		for path_name in class_paths:
			path = read_obj("{}/{}/{}".format(test_dir, video_class, path_name))
			path.normalize()

			classifications = classifyDTW(training_data, path)
			update_statistics(statistics, classifications, video_class)

	# Record new paths and classify them for video clips without existing path files.
	print("Generating paths for new files...")
	for video_class in video_tree:
		class_videos = video_tree[video_class]
		for video_name in class_videos:
			# Ignore videos which already have a path associated
			if "{}.path".format(video_name) in path_tree[video_class]:
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

			video_data = tracepath_from_frames(video, height=height, fps=fps, tracker=tracker)
			video_data.normalize()
			write_obj("{}/{}/{}".format(test_dir, video_class, "{}.path".format(video_name)), video_data)

			classifications = classifyDTW(training_data, video_data, video_class)
			update_statistics(statistics, classifications, video_class)

	print_statistics(statistics)

if __name__ == "__main__":
	predict()
