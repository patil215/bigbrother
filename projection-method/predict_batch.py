import itertools
import math
import os
import time
from collections import defaultdict
from time import sleep
import sys

import click
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

from classify import bfs_segment, classifyDTW, get_class_time_ranges, prep_data, new_prediction
from fileutils import (get_test_path_tree, get_test_segment_tree, read_obj,
                       read_training_data, write_obj)
from project import eulerAnglesToRotationMatrix, estimate_paper_rotation
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

def print_distr(statistics):
	keys = sorted(statistics.keys())
	print(
		tabulate(
			[[statistics[k] for k in keys]],
			headers=keys
		)
	)

def plot_distr(statistics, length):
	plt.plot(range(1, 11), [statistics["correct_" + str(i)] / statistics["total"] for i in range(1, 11)])
	plt.plot(range(1, 11), [(i / 10)**length for i in range(1, 11)])
	plt.show()

def update_statistics_batch(statistics, classifications, video_class):
	classes = video_class.split("_")

	for i in range(1, 11):
		if "correct_" + str(i) not in statistics:
			statistics["correct_" + str(i)] = 0

		correct = True
		for preds, actual in zip(classifications, classes):
			if actual not in preds[:i]:
				correct = False
				break

		if correct:	
			statistics["correct_" + str(i)] += 1
	statistics["total"] += 1
	print_distr(statistics)


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


def do_prediction(training_data, path, statistics, video_class):
	classifications = classifyDTW(training_data, path)
	update_statistics(statistics, classifications, video_class)


def do_prediction_batch(training_data, path, sequence_length, statistics, video_class):
	print(video_class)
	prediction = new_prediction(path, training_data, sequence_length)
	print(prediction)
	print()
	update_statistics_batch(statistics, prediction, video_class)


@click.command()
@click.argument('test_dir')
@click.option('-d', '--data', help="Location of the data directory", default="data")
@click.option('-a', '--angle', help="Camera position in degrees", nargs=3, default=None)
@click.option('-f', '--frame', help="Representative frame used for estimating angle", default=None)
@click.option('-l', '--length', help="Digit sequence length", default=1)
def predict(test_dir, data, angle, frame, length):
	if not os.path.exists(test_dir):
		print("Invalid test directory provided!")
		return
	if not os.path.exists(data):
		print("Invalid data directory provided!")
		return

	path_tree = get_test_path_tree(test_dir)
	training_data = read_training_data(data)

	if angle:
		print("Using provided angle")
	elif frame:
		# Automatically estimate angle
		angle = estimate_paper_rotation(frame)
		print("Autodetecting angles: got {}".format(angle))
	else:
		print("Either supply an angle or a frame")
		sys.exit(1)

	x, y, z = [math.radians(int(d)) for d in angle]
	transform = eulerAnglesToRotationMatrix(np.array([x, y, z]))
	prep_data(training_data, transform)

	if length == 1:
		statistics = defaultdict(lambda: defaultdict(int))
	else:
		statistics = defaultdict(int)

	# Classify test paths.
	print("Classifying test paths...")
	for video_class in path_tree:
		class_paths = path_tree[video_class]
		for path_name in class_paths:
			path = read_obj(
				"{}/{}/{}".format(test_dir, video_class, path_name))
			path.normalize()
			#path.interpolate(50)

			#predict_space_frames(video_class, inverse_path, length)
			if length == 1:
				do_prediction(training_data, path, statistics, video_class)
			else:
				do_prediction_batch(training_data, path, length, statistics, video_class)

	if length == 1:
		print_statistics(statistics)
	else:
		plot_distr(statistics, length)
		print(statistics)


if __name__ == "__main__":
	predict()
