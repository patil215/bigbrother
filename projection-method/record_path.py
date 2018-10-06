import argparse
import cv2
import time
import click
import numpy as np
from tracepoint import TracePoint, TracePath
import pickle
import os
from vizutils import draw_tracepoints
from fileutils import write_obj, read_obj

def create_blank(width, height, rgb_color=(0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = rgb_color
    return image

def current_time_millis():
    return int(round(time.time() * 1000))

def handle_mouse_move(event, x, y, flag, params):
    path = params[0]
    start_time = params[1]
    path.add(TracePoint((x, y, 0), current_time_millis() - start_time))

@click.command()
@click.argument('recordname')
@click.option('-c', '--count', help="Number of data samples to record", default=1)
@click.option('-s', '--start', help="Start by labelling with this index", default=0)
@click.option('-f', '--folder', help="Folder to save in (by default, data/[recordname]")
def record(recordname, count, start, folder):
    folder = 'data/' + recordname if not folder else folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(count):
        canvas = create_blank(512, 512, rgb_color=(0, 0, 0))
        path = TracePath()
        start_time = None
        recording = False

        while True:
            cv2.imshow("canvas", canvas)
            if recording:
                draw_tracepoints(canvas, path)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                if recording:
                    recording = False
                    cv2.setMouseCallback("canvas", lambda *args : None)
                    break
                else:
                    path = TracePath()
                    recording = True
                    start_time = current_time_millis()
                    cv2.setMouseCallback('canvas', handle_mouse_move, [path, start_time])

        write_obj(folder + '/' + recordname + '-' + str(start + i), path)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    record()




