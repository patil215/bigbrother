import cv2
import imutils

class Tracker:

	def __init__(self, frame, tracker_type='CSRT', height=700, bbox=None):
		"""bbox = bounding box picked out of the image"""
		self.tracker = self.pickTracker(tracker_type)
		self.height = height

		print("Original image has dimensions " + str(frame.shape))
		scale = frame.shape[0] / height
		print("Scale to show is " + str(scale))

		# No bbox, so prompt for it
		if bbox == None:
			bbox = cv2.selectROI("Select ROI", imutils.resize(frame, height=height), False)
			cv2.destroyWindow("Select ROI")
			cv2.waitKey(1)

			print("Old bbox is " + str(bbox))
			new_bbox = (int(bbox[0] * scale), int(bbox[1] * scale), 
						int(bbox[2] * scale), int(bbox[3] * scale))
			bbox = new_bbox
			print("New bbox is " + str(bbox))

		self.bbox = bbox
		ok = self.tracker.init(frame, bbox)

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