import cv2
import imutils

class Tracker:

	def __init__(self, frame, tracker_type, height=800, bbox=None):
		"""bbox = bounding box picked out of the image"""
		self.tracker = self.pickTracker(tracker_type)

		print("Original image has dimensions " + str(frame.shape))
		scale = frame.shape[0] / height
		print("Scale to show is " + str(scale))

		# No bbox, so prompt for it
		if bbox == None:
			bbox = cv2.selectROI(imutils.resize(frame, height=height), False)
			print("Old bbox is " + str(bbox))
			new_bbox = (bbox[0] * scale, bbox[1] * scale, bbox[2] * scale, bbox[3] * scale)
			bbox = new_bbox
			print("New bbox is " + str(bbox))

		ok = self.tracker.init(frame, bbox)

	def track(self, frame, display=False):
		ok, bbox = self.tracker.update(frame)
		if ok:
			if display:
				p1 = (int(bbox[0]), int(bbox[1]))
				p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
				cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
				frame = imutils.resize(frame, height=700)
				cv2.imshow("Tracking", frame)

			return bbox
		else:
			print("Not OK")
			raise ValueError("bad bbox")

	def pickTracker(self, tracker_type):
		if tracker_type == 'BOOSTING':
			tracker = cv2.TrackerBoosting_create()
		if tracker_type == 'MIL':
			tracker = cv2.TrackerMIL_create()
		if tracker_type == 'KCF':
			tracker = cv2.TrackerKCF_create()
		if tracker_type == 'TLD':
			tracker = cv2.TrackerTLD_create()
		if tracker_type == 'MEDIANFLOW':
			tracker = cv2.TrackerMedianFlow_create()
		if tracker_type == 'GOTURN':
			tracker = cv2.TrackerGOTURN_create()
		if tracker_type == 'MOSSE':
			tracker = cv2.TrackerMOSSE_create()
		if tracker_type == "CSRT":
			tracker = cv2.TrackerCSRT_create()

		return tracker