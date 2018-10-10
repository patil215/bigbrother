import numpy as np
import math
import pickle
import cv2
from tracepoint import TracePoint
from vizutils import draw_tracepoints
from classify import classifyDTW
from fileutils import read_obj, write_obj

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


# Test character
"""for x in np.linspace(0, 2 * math.pi, 8):
    for y in np.linspace(0, 2 * math.pi, 8):
        for z in np.linspace(0, 2 * math.pi, 8):
            print(eulerAnglesToRotationMatrix(np.array([x, y, z])))"""


#transform = eulerAnglesToRotationMatrix(np.array([0, 7 * math.pi / 16, math.pi / 4]))
