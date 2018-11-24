import numpy as np
import math
import cv2

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

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles (degrees)
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])

def estimate_paper_rotation(image):
    reference_points = []
    def handle_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            reference_points.append((x, y))

    # Read Image
    im = cv2.imread(image);

    print("Pick the top left, top right, bottom right, and bottom left corners you see in order, then hit a key.")

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", handle_click)
    cv2.imshow("image", im)
    cv2.waitKey(0)

    if len(reference_points) < 4:
        print("Four points not marked, exiting")
        sys.exit(0)

    size = im.shape
         
    image_points = np.array(reference_points, dtype="double")
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Top left corner
                                (0.0, -1100.0, 0.0),        # Bottom left corner
                                (850.0, -1100.0, 0.0),     # Bottom right corner
                                (850.0, 0.0, 0.0)      # Top right corner
                            ])
     
     
    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
     
    print("Camera Matrix :\n {0}".format(camera_matrix))
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
     
    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))
    angles = rotationMatrixToEulerAngles(cv2.Rodrigues(rotation_vector)[0])
    print(angles)
    return (angles[0] * -1, angles[1] * -1, angles[2] * -1)
     