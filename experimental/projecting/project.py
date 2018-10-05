import numpy as np
import math

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

# Test character
for x in np.linspace(0, 2 * math.pi, 8):
    for y in np.linspace(0, 2 * math.pi, 8):
        for z in np.linspace(0, 2 * math.pi, 8):
            print(eulerAnglesToRotationMatrix(np.array([x, y, z])))
