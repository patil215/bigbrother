import cv2
import numpy as np

img = cv2.imread("./hello.jpg")
edges = cv2.Canny(img, 0, 150)
print(edges)
cv2.imshow("edges", edges)
cv2.waitKey(0)