import argparse
import cv2
import numpy as np

refPt = []

def click_and_transform(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
rows, cols = image.shape[:2]
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_transform)

cv2.imshow("image", image)
cv2.waitKey(0)

if len(refPt) == 4:
    src_pts = np.float32([refPt[0], refPt[1], refPt[3], refPt[2]])
    dst_pts = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    print(projective_matrix)
    img_output = cv2.warpPerspective(clone, projective_matrix, (cols, rows))
    cv2.imshow("output", img_output)
    cv2.waitKey(0)
else:
    print("Four points not marked, exiting")

cv2.destroyAllWindows()

