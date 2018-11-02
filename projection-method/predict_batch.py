import math
import os
import random
import time
from time import sleep

import click
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

from classify import classifyDTW, computeSegment, prepData
from fileutils import read_obj, readData, readVideosLazy, write_obj
from motiontrack import Tracker
from project import eulerAnglesToRotationMatrix
from readvideo import getTracePathFromFrames
from tracepoint import TracePath, TracePoint
from vizutils import draw_tracepoints, plotPath, request_bounding_box, derive_alt_bounding_boxes


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
	random.seed(1337)

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
			initial_frame = video[0]

			metadata_path = test_dir + '/' + video_class + '/.' + video_base_name + '.meta'
			base_bbox = read_obj(metadata_path)

			if not base_bbox:
				base_bbox = request_bounding_box(initial_frame, height=height)
				write_obj(metadata_path, base_bbox)

			bboxes = [base_bbox] + derive_alt_bounding_boxes(initial_frame, base_bbox, qty=4, fuzz=5)
			video_data = []
			for bbox in bboxes:
				path = getTracePathFromFrames(video, fps=fps, tracker=Tracker(initial_frame, bbox=bbox))
				path.normalize()
				video_data.append(path)

			classifications = [classifyDTW(data, video_datum)[0][0] for video_datum in video_data]
			top_class = max(set(classifications), key=classifications.count)
			print("Truth: {} Guesses: {}".format(video_class, classifications))

			if top_class == video_class:
				correct += 1

			# video_data = getTracePathFromFrames(video, height=height, fps=fps, tracker=tracker)
			# video_data.normalize()
			# actual = classifyDTW(data, video_data)

			# top3 = [thing[0] for thing in actual[:3]]
			# top5 = [thing[0] for thing in actual[:5]]
			# print("Expected: {} Actual: {}".format(video_class, actual[0][0]))
			# if actual[0][0] == video_class:
			# 	correct += 1
			# if video_class in top3:
			# 	correct_top3 += 1
			# if video_class in top5:
			# 	correct_top5 += 1
			total += 1

	print("Correct: {} Correct (top 3): {} Correct (top 5): {} Total: {}".format(correct, correct_top3, correct_top5, total))


if __name__ == "__main__":
	predict()
