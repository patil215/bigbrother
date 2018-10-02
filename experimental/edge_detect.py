import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

# https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
img = cv2.imread(args["image"])
edges = cv2.Canny(img, 0, 150)
print(edges)

callback = lambda x : None

cv2.namedWindow("edges")

cv2.createTrackbar("min", "edges", 0, 500, callback)
cv2.createTrackbar("max", "edges", 150, 500, callback)
cv2.imshow("edges", edges)

while True:
    edges = cv2.Canny(img,
        cv2.getTrackbarPos("min", "edges"),
        cv2.getTrackbarPos("max", "edges"))
    
    cv2.imshow("edges", edges)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

