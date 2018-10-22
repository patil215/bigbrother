from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtw import dtw
from numpy.linalg import norm
import operator

def computeDTWDistance(x_actual, y_actual, x_test, y_test):
	dist_x, cost_x, acc_x, path_x = dtw(x_actual, x_test, dist=lambda x, y: abs(x - y))
	dist_y, cost_y, acc_y, path_y = dtw(y_actual, y_test, dist=lambda x, y: abs(x - y))

	distance = (dist_x + dist_y)**2
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
	results = {}
	for name in candidates.keys():
		candidate_path = candidates[name][0]

		minDist = min([computeDTWDistance(x_actual, y_actual, candidate.sequence(0), candidate.sequence(1))
			for candidate in candidates[name]])
		results[name] = minDist

	sorted_distances = sorted(results.items(), key=operator.itemgetter(1))
	printScores(sorted_distances)
	return sorted_distances[0][0]