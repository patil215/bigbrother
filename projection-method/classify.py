from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw
from dtw import dtw
from numpy.linalg import norm
import operator
from tracepoint import TracePath
import math

def get_class_time_ranges(data):
	ranges = {}
	for class_name in data:
		start_millis = min([tracepath.path[-1].t - tracepath.path[0].t for tracepath in data[class_name]])
		end_millis = max([tracepath.path[-1].t - tracepath.path[1].t for tracepath in data[class_name]])
		ranges[class_name] = (start_millis, end_millis)
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
	frames_per_step = int(20 / MILLIS_PER_FRAME)

	for end_frame_index in range(interval_start_frame, interval_end_frame, frames_per_step):
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

def bfs_segment(tracepath, candidates, num_digits, fps, 
		K=1,
		SPACE_START_MILLIS=500,
		SPACE_END_MILLIS=1000,
		SPACE_STEP_MILLIS=100
	):
	# Format: List[(total_score, class_name, frame_start, frame_end)]
	current_classifications = [[(0.0, "start", 0, 0)]]
	num_digits_sequenced = 0
	MILLIS_PER_FRAME = 1000.0 / fps

	class_time_ranges = get_class_time_ranges(candidates)

	num_digits = num_digits * 2 - 1  # Add room for spaces

	while num_digits_sequenced < num_digits:
		new_classifications = []
		for classification_list in current_classifications:
			latest_score, latest_class, latest_frame_start_index, latest_frame_end_index = classification_list[-1]

			classes_to_consider = []
			if num_digits_sequenced % 2 == 1:
				classes_to_consider = ["space"]
			else:
				classes_to_consider = list(candidates.keys())
				classes_to_consider.remove("space")

			for class_name in classes_to_consider:
				new_intervals = generate_intervals(latest_frame_end_index, class_time_ranges[class_name], len(tracepath.path), fps)
				for start_index, end_index in new_intervals:
					path_slice = TracePath(path=tracepath.path[start_index:end_index + 1])
					path_slice.normalize()
					candidates_to_consider = {class_name: candidates[class_name]}
					result, distance = classifyDTW(candidates_to_consider, path_slice)[0]

					new_classification_list = classification_list + [(latest_score + distance, result, start_index, end_index)]
					new_classifications.append(new_classification_list)

		# Take the top K
		current_classifications = sorted(new_classifications, key=lambda classification_list: classification_list[-1])[:K]
		print_classifications(current_classifications)
		num_digits_sequenced += 1

	return current_classifications

def prep_data(data, R):
	for category in data:
		for tracepath in data[category]:
			tracepath.transform(R)
			tracepath.normalize()

def recursive_segment(tracepath, candidates, num_digits_left, current_path_index=0, STEP_DURATION=50, FPS=29.97):
	if num_digits_left == 0:
		return [(0, [])]

	index_segments = []
	MILLIS_PER_FRAME = 1000 / FPS
	for end_index in range(current_path_index + int(400 / MILLIS_PER_FRAME), current_path_index + int(1000 / MILLIS_PER_FRAME), int((STEP_DURATION / 1000 * FPS))):
		if end_index >= len(tracepath.path):
			break
		index_segments.append((current_path_index, end_index))

	children_results = []
	for index_segment in index_segments:
		# classify
		path_slice = TracePath(path=tracepath.path[index_segment[0]:index_segment[1] + 1])
		path_slice.normalize()

		# Do our comparing and append the right one
		if num_digits_left % 2 == 0:
			result, distance = ("space", 0)
		else:
			candidates_to_consider = {}
			for key in candidates:
				if key != "space":
					candidates_to_consider[key] = candidates[key]
			result, distance = classifyDTW(candidates_to_consider, path_slice)[0]

		#print("{} {} {}".format(num_digits_left, result, distance))

		# Get top 10 of next recursive indices
		children = recursive_segment(tracepath, candidates, num_digits_left - 1, index_segment[1] + 1)
		for child in children:
			children_results.append((distance + child[0], [result] + child[1]))

	return sorted(children_results)[:10]

def compute_segment(tracepath, candidates, num_digits):
	return recursive_segment(tracepath, candidates, num_digits * 2 - 1)

def computeDTWDistance(x_actual, y_actual, x_test, y_test):
	dist_x, cost_x, acc_x, path_x = dtw(x_actual, x_test, dist=lambda x, y: abs(x - y))
	dist_y, cost_y, acc_y, path_y = dtw(y_actual, y_test, dist=lambda x, y: abs(x - y))

	distance = math.sqrt(dist_x ** 2 + dist_y ** 2)
	return distance

def printScores(sorted_distances):
	for item in sorted_distances:
		print(item[0] + ": " + str(item[1]))

def classifyDTW(candidates, path):
	"""Uses Dynamic Time Warping to classify a path as one of the candidates.
	Candidates: dict from class name to list of normalized TracePaths, ex) {"zero": [pathName]}
	path: normalized TracePath, what we're trying to classify.
	"""
	x_actual = path.sequence(0)
	y_actual = path.sequence(1)
	#results = []
	results = {}
	for name in candidates.keys():
		minDist = min([computeDTWDistance(x_actual, y_actual, candidate.sequence(0), candidate.sequence(1))
			for candidate in candidates[name]])
		results[name] = minDist

		#for candidate in candidates[name]:
			#dist = computeDTWDistance(x_actual, y_actual, candidate.sequence(0), candidate.sequence(1))
			#results.append((dist, name))

	sorted_distances = sorted(results.items(), key=operator.itemgetter(1))
	# printScores(sorted_distances)

	#sorted_distances = sorted(results)
	return sorted_distances
	#return (sorted_distances[0][0], sorted_distances[0][1])