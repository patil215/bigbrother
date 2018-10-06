import numpy as np
import math
import pickle
import cv2
from tracepoint import TracePoint
from vizutils import draw_tracepoints
from classify import classifyDTW
from fileutils import read_obj, write_obj
import matplotlib.pyplot as plt

# Calculates Rotation Matrix given euler angles.
# Theta is a 3D vector with X, Y, and Z rotation amounts (in degrees)
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

def create_blank(width, height, rgb_color=(0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = rgb_color
    return image

def drawTransformed(R, path):
    canvas = create_blank(512, 512, rgb_color=(0, 0, 0))
    path.transform(R)

    draw_tracepoints(canvas, path, fit_canvas=True)

    cv2.imshow("canvas", canvas)
    cv2.waitKey()

def plotPath(path, coordinate, color):
    pts = path.time_sequence(coordinate)
    plt.plot([p[0] for p in pts], [p[1] for p in pts], color)

def prep(path, R):
    path.transform(R)
    #path.normalize()

# Test character
"""for x in np.linspace(0, 2 * math.pi, 8):
    for y in np.linspace(0, 2 * math.pi, 8):
        for z in np.linspace(0, 2 * math.pi, 8):
            print(eulerAnglesToRotationMatrix(np.array([x, y, z])))"""

# drawTransformed(, path)

transform = eulerAnglesToRotationMatrix(np.array([math.pi / 4, math.pi / 4, math.pi / 8]))
#transform = eulerAnglesToRotationMatrix(np.array([0, 0, 0]))
path_zero = read_obj('data/zero/zero-0')
path_one = read_obj('data/one/one-0')
path_test = read_obj('data/zero/zero-1')


prep(path_zero, transform)
prep(path_one, transform)
prep(path_test, transform)

plotPath(path_zero, 0, 'r')
plotPath(path_one, 0, 'g')
plotPath(path_test, 0, 'b')

plotPath(path_zero, 1, 'r')
plotPath(path_one, 1, 'g')
plotPath(path_test, 1, 'b')

plt.show()

print(classifyDTW({"zero": path_zero, "one": path_one}, path_test))