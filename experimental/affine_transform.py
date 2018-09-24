import cv2
import numpy as np

img = cv2.imread('./hello.jpg')
rows, cols = img.shape[:2]

# We just need three points to get the affine transformation matrix
# Pick the origin (top left), upper right, and bottom left
src_points = np.float32([[0,0], [cols - 1, 0], [0, rows - 1]])

# Map these to origin, 0.6 * width on top row, 0.4 * width on bottom row
dst_points = np.float32([[0,0], [int(0.6 * (cols - 1)), 0], [int(0.4 * (cols - 1)), rows - 1]])
affine_matrix = cv2.getAffineTransform(src_points, dst_points)
img_output = cv2.warpAffine(img, affine_matrix, (cols, rows))

cv2.imshow('input', img)
cv2.imshow('output', img_output)
cv2.waitKey()

