from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw
from dtw import dtw
from numpy.linalg import norm
import operator
from tracepoint import TracePath
import math

def prepData(data, R):
	for category in data:
		for tracepath in data[category]:
			tracepath.transform(R)
			tracepath.normalize()


def recursiveSegment(tracepath, candidates, num_digits_left, current_path_index=0, STEP_DURATION=50, FPS=29.97):
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
		candidates_to_consider = {}
		if num_digits_left % 2 == 0:
			candidates_to_consider = {"space": candidates["space"]}
		else:
			for key in candidates:
				if key != "space":
					candidates_to_consider[key] = candidates[key]
		result, distance = classifyDTW(candidates_to_consider, path_slice)
		print("{} {} {}".format(num_digits_left, result, distance))

		# Get top 10 of next recursive indices
		children = recursiveSegment(tracepath, candidates, num_digits_left - 1, index_segment[1] + 1)
		for child in children:
			children_results.append((distance + child[0], [result] + child[1]))

	return sorted(children_results)[:10]

def computeSegment(tracepath, candidates, num_digits):
	return recursiveSegment(tracepath, candidates, num_digits)

def computeDTWDistance(x_actual, y_actual, x_test, y_test):
	dist_x, cost_x, acc_x, path_x = dtw(x_actual, x_test, dist=lambda x, y: abs(x - y))
	dist_y, cost_y, acc_y, path_y = dtw(y_actual, y_test, dist=lambda x, y: abs(x - y))

	distance = math.sqrt(dist_x ** 2 + dist_y ** 2)
	return distance
	#distance = abs(dist_x) + abs(dist_y)
	#return distance

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
		if name == 'space':
			continue

		minDist = min([computeDTWDistance(x_actual, y_actual, candidate.sequence(0), candidate.sequence(1))
			for candidate in candidates[name]])
		results[name] = minDist

		#for candidate in candidates[name]:
			#dist = computeDTWDistance(x_actual, y_actual, candidate.sequence(0), candidate.sequence(1))
			#results.append((dist, name))

	sorted_distances = sorted(results.items(), key=operator.itemgetter(1))
	printScores(sorted_distances)

	#sorted_distances = sorted(results)
	return sorted_distances
	#return (sorted_distances[0][0], sorted_distances[0][1])