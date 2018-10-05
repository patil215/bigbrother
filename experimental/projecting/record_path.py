import argparse
import cv2
import time

def current_time_millis():
    return int(round(time.time() * 1000))

class TracePoint:
    # T is in milliseconds since start of path trace.
    def __init__(self, x, y, z=0, t=0):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

canvas = np.zeros((512, 512, 3))

path = []
start_time = None
recording = False

def mouseCallback(event, x, y, flag, params):
    if recording:
        path.append(TracePoint(x, y, 0, current_time_millis - start_time))

while True:
    cv2.imshow("canvas", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if recording:
            recording = False
            break
        else:
            path = []
            recording = True

    if recording:
        path.append(TracePoint(


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)
