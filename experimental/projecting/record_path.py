import argparse
import cv2
import time
import click
import numpy as np

class TracePoint:
    # t signifies milliseconds since start of path trace.
    def __init__(self, x, y, z=0, t=0):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def pos(self):
        return (self.x, self.y)

def create_blank(width, height, rgb_color=(0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = rgb_color
    return image

def current_time_millis():
    return int(round(time.time() * 1000))

canvas = create_blank(512, 512, rgb_color=(0, 0, 0))

path = []
start_time = None
recording = False
param = None

def handle_mouse_move(event, x, y, flag, params):
    param = (x, y)
    if recording:
        path.append(TracePoint(x, y, 0, current_time_millis() - start_time))

while True:
    if recording:
        # Draw line segments connecting points
        for index in range(len(path) - 1):
            cv2.line(canvas, pts[index].pos, pts[index + 1].pos, (255, 255, 255))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        if recording:
            recording = False
            break
        else:
            path = []
            recording = True
            cv2.setMouseCallback('Drawing spline', handle_mouse_move, param)

    cv2.imshow("canvas", canvas)

# Dump points to file


cv2.destroyAllWindows()
