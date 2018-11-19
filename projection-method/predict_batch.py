import itertools
import math
import os
import time
from collections import defaultdict
from time import sleep

import click
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

from classify import bfs_segment, classifyDTW, get_class_time_ranges, prep_data
from fileutils import (get_test_path_tree, get_test_segment_tree, read_obj,
                       read_training_data, write_obj)
from project import eulerAnglesToRotationMatrix
from tracepoint import TracePath, TracePoint
from vizutils import draw_tracepoints, plotPath


def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()


def update_statistics(statistics, classifications, video_class):
	top3 = [thing[0] for thing in classifications[:3]]
	top5 = [thing[0] for thing in classifications[:5]]
	print("Predicted: {}, Expected: {}".format(
		classifications[0][0], video_class))

	if statistics[video_class]["classification"] == 0:
		statistics[video_class]["classification"] = defaultdict(int)
	statistics[video_class]["classification"][classifications[0][0]] += 1

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
			[[c, statistics[c]["correct"]/statistics[c]["total"], statistics[c]["correct_top3"],
				statistics[c]["correct_top5"], statistics[c]["total"]] for c in statistics],
			headers=["Class", "Percent correct", "Top 3", "Top 5", "Total"]
		)
	)

	class_names = list(statistics.keys())
	pred = []
	truth = []

	# generate samples for the confusion matrix
	for truth_class in class_names:
		for prediction_class in statistics[truth_class]["classification"]:
			for i in range(statistics[truth_class]["classification"][prediction_class]):
				pred.append(prediction_class)
				truth.append(truth_class)

	cnf_matrix = confusion_matrix(truth, pred, labels=class_names)
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
						title='Confusion matrix, without normalization')

	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
						title='Normalized confusion matrix')

	plt.show()

def find_greatest_speed_index(path, valid_indices):
	greatest_speed = 0
	greatest_speed_index = 0
	for i in range(len(path) - 1):
		velocity_x = path[i + 1].pos[0] - path[i].pos[0]
		velocity_y = path[i + 1].pos[1] - path[i].pos[1]
		speed = math.sqrt(velocity_x**2 + velocity_y**2)
		if speed > greatest_speed and i in valid_indices:
			greatest_speed = speed
			greatest_speed_index = i

	return greatest_speed_index

def find_lowest_speed_indices(path, range_inds):
	speeds = [] # Tuple of (speed, index)
	for i in range(range_inds[0], range_inds[1]):
		velocity_x = path[i + 1].pos[0] - path[i].pos[0]
		velocity_y = path[i + 1].pos[1] - path[i].pos[1]
		speed = math.sqrt(velocity_x**2 + velocity_y**2)
		speeds.append((speed, i))
	return [p[1] for p in sorted(speeds)[:2]]


def find_space_frames(path, num_to_find):
	space_frames = []
	valid_indices = set([i for i in range(len(path))])
	while len(space_frames) < num_to_find:
		greatest_speed_index = find_greatest_speed_index(path, valid_indices)
		for i in range(greatest_speed_index - 15, greatest_speed_index0.005099617485316769, 75 + 15): # TODO don't hardcode frames
			if i in valid_indices:
				valid_indices.remove(i)

		start_low_range = greatest_speed_index + 15
		end_low_range = min(len(path), greatest_speed_index + 25)
		space_frames.append((greatest_speed_index - 3, find_lowest_speed_indices(path, (start_low_range, end_low_range))))
	return sorted(space_frames)


def do_prediction(training_data, path, sequence_length, statistics, video_class):
	if sequence_length == 1:
		classifications = classifyDTW(training_data, path)
		update_statistics(statistics, classifications, video_class)
	else:
		print(video_class)
		#space_frames = find_space_frames(path.path, sequence_length - 1)
		#print(space_frames)
		print(sorted(list(path.checkpoint_indices)))
		#return

		print(sorted(list(path.checkpoint_indices)))
		bfs_segment(path, training_data, sequence_length)


@click.command()
@click.argument('test_dir')
@click.option('-d', '--data', help="Location of the data directory", default="data")
@click.option('-a', '--angle', help="Camera position in degrees", nargs=3, default=(0, 0, 0))
@click.option('-l', '--length', help="Digit sequence length", default=1)
def predict(test_dir, data, angle, length):
	if not os.path.exists(test_dir):
		print("Invalid test directory provided!")
		return
	if not os.path.exists(data):
		print("Invalid data directory provided!")
		return

	path_tree = get_test_path_tree(test_dir)
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
			path = read_obj(
				"{}/{}/{}".format(test_dir, video_class, path_name))
			path.normalize()

			do_prediction(training_data, path, length, statistics, video_class)

	print_statistics(statistics)


if __name__ == "__main__":
	predict()
