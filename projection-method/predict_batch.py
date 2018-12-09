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


def update_classifications(classifications, prediction, video_class):
	top_1_predicted = "_".join([p[0] for p in prediction])
	classifications[video_class][top_1_predicted] += 1

def print_distr(statistics):
	keys = sorted(statistics.keys())
	print(
		tabulate(
			[[statistics[k] for k in keys]],
			headers=keys
		)
	)

def plot_distr(statistics, length):
	plt.plot(range(1, 11), [statistics[str(i)] / statistics["total"] for i in range(1, 11)])
	plt.plot(range(1, 11), [(i / 10)**length for i in range(1, 11)])
	plt.show()

def print_average_rank(statistics):
	total = statistics["total"]

	for i in range(0, 11):
		for j in range(i + 1, 11):
			statistics[str(j)] -= statistics[str(i)]

	del statistics["total"]
	print("Average rank: {}".format(sum([statistics[key] * int(key) for key in statistics]) / total))

def update_statistics(statistics, classifications, video_class):
	classes = video_class.split("_")

	for i in range(1, 11):
		if str(i) not in statistics:
			statistics[str(i)] = 0

		correct = True
		for preds, actual in zip(classifications, classes):
			if actual not in preds[:i]:
				correct = False
				break

		if correct:	
			statistics[str(i)] += 1
	statistics["total"] += 1
	print_distr(statistics)


def output_classifications(classifications):

	def compute_percentage(actual, classifications):
		total = sum([classifications[actual][p] for p in classifications[actual]])
		return (classifications[actual][actual] / total) * 100

	print("BY CLASS:")
	print(
		tabulate(
			[[c, compute_percentage(c, classifications)] for c in classifications],
			headers=["Class", "Percent correct"]
		)
	)

	class_names = list(classifications.keys())
	preds = []
	truths = []
	# Generate samples for the confusion matrix
	for truth_class in class_names:
		for prediction_class in classifications[truth_class]:
			for _ in range(classifications[truth_class][prediction_class]):
				preds.append(prediction_class)
				truths.append(truth_class)

	cnf_matrix = confusion_matrix(truths, preds, labels=class_names)
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


def do_prediction(training_data, path, sequence_length, statistics, classifications, video_class):
	try:
		prediction = new_prediction(path, training_data, sequence_length)
		print("Actual: {}, Predictions: {}".format(video_class, prediction))
		update_statistics(statistics, prediction, video_class)
		update_classifications(classifications, prediction, video_class)
		print()
	except Exception as e:
		print("Skipping")

def test_batch(test_dir, data, angle, length):
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

	statistics = defaultdict(int)
	classifications = defaultdict(lambda: defaultdict(int)) # (actual, predicted) eg. classifications[zero][one] = 1

	# Classify test paths.
	print("Classifying test paths...")
	for video_class in path_tree:
		class_paths = path_tree[video_class]
		for path_name in class_paths:
			path = read_obj(
				"{}/{}/{}".format(test_dir, video_class, path_name))
			path.normalize()
			print(video_class + " " + path_name)

			do_prediction(training_data, path, length, statistics, classifications, video_class)

	return (statistics, classifications)


@click.command()
@click.argument('test_dir')
@click.option('-d', '--data', help="Location of the data directory", default="data")
@click.option('-a', '--angle', help="Camera position in degrees", nargs=3, default=None)
@click.option('-f', '--frame', help="Representative frame used for estimating angle", default=None)
@click.option('-l', '--length', help="Digit sequence length", default=1)
def predict(test_dir, data, angle, frame, length):
	if angle:
		print("Using provided angle")
	elif frame:
		# Automatically estimate angle
		angle = estimate_paper_rotation(frame)
		print("Autodetecting angles: got {}".format(angle))
	else:
		print("Either supply an angle or a frame")
		sys.exit(1)

	statistics, classifications = test_batch(test_dir, data, angle, length)

	plot_distr(statistics, length)
	print_average_rank(statistics)
	if length == 1:
		output_classifications(classifications)

if __name__ == "__main__":
	predict()
