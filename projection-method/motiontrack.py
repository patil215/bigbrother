import cv2
import imutils
from vizutils import request_bounding_box

class Tracker:

	def __init__(self, frame, tracker_type='CSRT', height=700, bbox=None):
		"""bbox = bounding box picked out of the image"""
		self.tracker = self.pickTracker(tracker_type)
		self.height = height
		if bbox is not None:
			self.bbox = bbox
		else:
			self.bbox = request_bounding_box(frame, height)

		ok = self.tracker.init(frame, self.bbox)

	def track(self, frame, display=False):
		ok, bbox = self.tracker.update(frame)
		if ok:
			if display:
				p1 = (int(bbox[0]), int(bbox[1]))
				p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
				cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
				frame = imutils.resize(frame, height=self.height)
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