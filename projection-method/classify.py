from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtw import dtw
from numpy.linalg import norm

def classifyDTW(candidates, path):
	"""Uses Dynamic Time Warping to classify a path as one of the candidates.
	Candidates: dict from class name to list of normalized TracePaths, ex) {"zero": [pathName]}
	path: normalized TracePath, what we're trying to classify.
	"""
	x_actual = path.sequence(0)
	y_actual = path.sequence(1)
	minDistance = 1000000000000000000000000000000
	bestCandidate = None
	for name in candidates.keys():
		candidate_path = candidates[name][0]
		x_test = candidate_path.sequence(0)
		y_test = candidate_path.sequence(1)
		#distance_x, _ = fastdtw(x_actual, x_test, dist=euclidean)
		#distance_y, _ = fastdtw(y_actual, y_test, dist=euclidean)
		print(x_actual)
		print(x_test)
		dist_x, cost_x, acc_x, path_x = dtw(x_actual, x_test, dist=lambda x, y: abs(x - y))
		dist_y, cost_y, acc_y, path_y = dtw(y_actual, y_test, dist=lambda x, y: abs(x - y))

		distance = (dist_x + dist_y)**2
		print(name + ": " + str(dist_x) + " " + str(dist_y))

		if distance < minDistance:
			bestCandidate = name
			minDistance = distance

	return bestCandidate