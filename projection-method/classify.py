from scipy.spatial.distance import euclidean
from dtw import dtw
from numpy.linalg import norm
import operator
from tracepoint import TracePath
import math
import matplotlib.pyplot as plt
from numpy import std

def predict_space_frames(video_class, path, sequence_length):
	space_frames = find_space_frames(path.path, sequence_length - 1)
	print(space_frames)

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
		for i in range(greatest_speed_index - 15, greatest_speed_index + 15): # TODO don't hardcode frames
			if i in valid_indices:
				valid_indices.remove(i)

		start_low_range = greatest_speed_index + 15
		end_low_range = min(len(path), greatest_speed_index + 25)
		#space_frames.append((greatest_speed_index - 3, find_lowest_speed_indices(path, (start_low_range, end_low_range))))
		space_frames.append(greatest_speed_index - 3)
	return sorted(space_frames)

def new_prediction(tracepath, candidates, num_digits):
	space_beginning_frames = find_space_frames(tracepath.path, num_digits - 1)
	print("Space beginning frames are: {}".format(space_beginning_frames))

	classified_sequence = []
	# First digit we get for free
	for i in range(num_digits):
		if i == 0:
			# First digit from start to the first space, guaranteed
			possible_space_intervals = [(0, 0)]
			end_frame = space_beginning_frames[i]
		elif i == num_digits - 1:
			# Last digit from space to end of tracepath
			possible_space_intervals = generate_intervals(space_beginning_frames[i - 1], (200, 600), len(tracepath.path), tracepath.fps())
			end_frame = len(tracepath.path)
		else:
			# Guess that the space is about 25 frames, and beginning of digit starts after
			possible_space_intervals = generate_intervals(space_beginning_frames[i - 1], (200, 600), len(tracepath.path), tracepath.fps())
			end_frame = space_beginning_frames[i]

		results = []
		for interval in possible_space_intervals:
			start_frame = interval[1]

			path_slice = TracePath(path=tracepath.path[start_frame:end_frame + 1])
			path_slice.normalize()
			preds = classifyDTW(candidates, path_slice, include_space=False)[:3]
			for class_name, distance in preds:
				results.append((distance, class_name))

		results = sorted(results)
		initial_len = len(results)
		unique_classes = set()
		while len(unique_classes) < 3 and len(unique_classes) < initial_len:
			unique_classes.add(results[0][1])
			results = results[1:]
		classified_sequence.append(list(unique_classes))

	return classified_sequence


def get_class_time_ranges(data, z_index=2.5):
	ranges = {}
	for class_name in data:
		start_millis = min([tracepath.path[-1].t - tracepath.path[0].t for tracepath in data[class_name]])
		end_millis = max([tracepath.path[-1].t - tracepath.path[0].t for tracepath in data[class_name]])
		average = (start_millis + end_millis) / 2
		stdev = std([tracepath.path[-1].t - tracepath.path[0].t for tracepath in data[class_name]])
		range_start = max(0, start_millis - (stdev * z_index))
		range_end = end_millis + (stdev * z_index)
		ranges[class_name] = (range_start, range_end)
	return ranges

def generate_intervals(
		start_frame_index,
		time_range,
		path_length,
		fps
	):
	MILLIS_PER_FRAME = 1000.0 / fps
	intervals = []

	interval_start_frame = start_frame_index + int(time_range[0] / MILLIS_PER_FRAME)
	interval_end_frame = start_frame_index + int(time_range[1] / MILLIS_PER_FRAME)
	frames_per_step = 1

	for end_frame_index in range(interval_start_frame, interval_end_frame, frames_per_step):
		if end_frame_index - start_frame_index == 0:
			continue
		if end_frame_index >= path_length:
			break  # Don't include segments going out of bounds
		intervals.append((start_frame_index, end_frame_index))

	return intervals

def print_classifications(classification_lists):
	print("Candidates:")
	for classification_list in classification_lists:
		total_score = classification_list[-1][0]
		classes = [x[1] for x in classification_list[1:]]
		times = [str((x[2], x[3])) for x in classification_list[1:]]
		print("{}: Times: {} (Score: {})".format("->".join(classes), ",".join(times), total_score))
	print()

def plot_interval_vs_score(candidates, class_name, intervals, tracepath):
	data = []
	print(intervals)
	for start_index, end_index in intervals:
		path_slice = TracePath(path=tracepath.path[start_index:end_index + 1])
		path_slice.normalize()
		candidates_to_consider = {class_name: candidates[class_name]}
		result, distance = classifyDTW(candidates_to_consider, path_slice)[0]
		data.append(((end_index - start_index), distance))

	plt.plot([x[0] for x in data], [x[1] for x in data])
	plt.ylabel("Score")
	plt.xlabel("Interval length (frames)")
	plt.title("Score for class {}".format(class_name))
	plt.show()

def bfs_segment(tracepath, candidates, num_digits, K=5, allow_growth=True):
	# Format: List[(total_score, class_name, frame_start, frame_end)]
	current_classifications = [[(0.0, "start", 0, 0)]]
	num_digits_sequenced = 0

	class_time_ranges = get_class_time_ranges(candidates)

	num_digits = num_digits * 2 - 1  # Add room for spaces

	while num_digits_sequenced < num_digits:
		new_classifications = []
		for classification_list in current_classifications:
			latest_score, latest_class, latest_frame_start_index, latest_frame_end_index = classification_list[-1]

			classes_to_consider = []
			include_space = False
			if num_digits_sequenced % 2 == 1:
				include_space = True
				classes_to_consider = ["space"]
			else:
				classes_to_consider = list(candidates.keys())
				classes_to_consider.remove("space")

			for class_name in classes_to_consider:
				new_intervals = generate_intervals(latest_frame_end_index, class_time_ranges[class_name], len(tracepath.path), tracepath.fps())
				for start_index, end_index in new_intervals:
					path_slice = TracePath(path=tracepath.path[start_index:end_index + 1])
					path_slice.normalize()
					candidates_to_consider = {class_name: candidates[class_name]}
					result, distance = classifyDTW(candidates_to_consider, path_slice, include_space=include_space)[0]

					new_classification_list = classification_list + [(latest_score + distance, result, start_index, end_index)]
					new_classifications.append(new_classification_list)

		# Take the top K
		sorted_classifications = sorted(new_classifications, key=lambda classification_list: classification_list[-1])

		# Implement affirmative action
		unique_classes = set()
		unique_classifications = list()
		for classification in sorted_classifications:
			classes = tuple([x[1] for x in classification[1:]])

			if classes not in unique_classes:
				unique_classes.add(classes)
				unique_classifications.append(classification)
			
			bound = K * (num_digits_sequenced + 1) if allow_growth else K
			if len(unique_classes) >= bound:
				break

		current_classifications = unique_classifications
		print_classifications(current_classifications)
		num_digits_sequenced += 1

	return current_classifications

def prep_data(data, R):
	for category in data:
		for tracepath in data[category]:
			tracepath.transform(R)
			tracepath.normalize()

def computeDTWDistance(x_actual, y_actual, x_test, y_test):
	dist_x, cost_x, acc_x, path_x = dtw(x_actual, x_test, dist=lambda x, y: abs(x - y))
	dist_y, cost_y, acc_y, path_y = dtw(y_actual, y_test, dist=lambda x, y: abs(x - y))

	distance = math.sqrt(dist_x ** 2 + dist_y ** 2)
	return distance

def printScores(sorted_distances):
	for item in sorted_distances:
		print(item[0] + ": " + str(item[1]))

def classifyDTW(candidates, path, time_penalty_factor=1250, include_space=False):
	"""Uses Dynamic Time Warping to classify a path as one of the candidates.
	Candidates: dict from class name to list of normalized TracePaths, ex) {"zero": [pathName]}
	path: normalized TracePath, what we're trying to classify.
	Higher time penalty factor means penalizing less.
	"""
	if not include_space and "space" in candidates:
		del candidates["space"]

	path_length = path.path[-1].t - path.path[0].t
	time_ranges = get_class_time_ranges(candidates)

	x_actual = path.sequence(0)
	y_actual = path.sequence(1)

	results = {}
	for name in candidates.keys():
		minDist = min([computeDTWDistance(x_actual, y_actual, candidate.sequence(0), candidate.sequence(1))
			for candidate in candidates[name]])

		candidate_range = time_ranges[name]
		penalty = 0
		if path_length < candidate_range[0]:
			penalty = abs(candidate_range[0] - path_length) / time_penalty_factor
		elif path_length > candidate_range[1]:
			penalty = abs(candidate_range[1] - path_length) / time_penalty_factor
		#minDist += penalty

		results[name] = minDist

	sorted_distances = sorted(results.items(), key=operator.itemgetter(1))
	return sorted_distances
