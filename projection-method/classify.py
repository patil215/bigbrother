from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def classifyDTW(candidates, path):
	"""Uses Dynamic Time Warping to classify a path as one of the candidates.
	Candidates: dict from class name to normalized TracePath, ex) {"zero": pathName}
	path: normalized TracePath, what we're trying to classify.
	"""
	x_actual = path.time_sequence(0)
	y_actual = path.time_sequence(1)
	minDistance = 1000000000000000000000000000000
	bestCandidate = None
	for name in candidates.keys():
		candidate_path = candidates[name][0]
		x_test = candidate_path.time_sequence(0)
		y_test = candidate_path.time_sequence(1)

		distance_x, _ = fastdtw(x_actual, x_test, dist=euclidean)
		distance_y, _ = fastdtw(y_actual, y_test, dist=euclidean)
		#distance = (distance_x + distance_y)**2
		distance = distance_y

		print(name)
		print(distance_x)
		print(distance_y)
		print("")

		if distance < minDistance:
			bestCandidate = name
			minDistance = distance

	return bestCandidate