import numpy as np
import cv2

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""

    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))

    # Fill image with color
    image[:] = color

    return image

def draw_tracepoints(frame, tracepoints, scale=1.0, fit_canvas=False, color=(255, 255, 255)):
	"""Given a list of tracepoints, draw the path on the given frame."""

	xs = np.array([point.pos[0] for point in tracepoints])
	ys = np.array([point.pos[1] for point in tracepoints])

	height = frame.shape[0]
	width = frame.shape[1]

	if fit_canvas:
		draw_points = list(zip(np.interp(xs, (xs.min(), xs.max()), (0, width)), np.interp(ys, (ys.min(), ys.max()), (0, height))))
	else:
		draw_points = list(zip(xs, ys))

	draw_points = list(map(lambda p: (int(p[0] * scale), int(p[1] * scale)), draw_points))

	if len(tracepoints) <= 1:
		return

	for i in range(len(tracepoints) - 1):
		cv2.line(frame, draw_points[i], draw_points[i + 1], color)