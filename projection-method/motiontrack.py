import cv2
import imutils
from vizutils import request_bounding_box

class Tracker:

	def __init__(self, frame, height, bbox=None):
		"""bbox = bounding box picked out of the image"""
		self.tracker = cv2.TrackerCSRT_create()
		self.height = height
		if bbox is not None:
			self.bbox = bbox
		else:
			self.bbox = request_bounding_box(frame, height)

		ok = self.tracker.init(frame, self.bbox)

	def track(self, frame, display=False):
		ok, bbox = self.tracker.update(frame)
		if not ok:
			print("Tracking failed. Not OK")
			raise ValueError("tracking failed, bad bbox")

		return bbox
