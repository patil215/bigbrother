import cv2
gray_img = cv2.imread('./hello.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Grayscale', gray_img)
cv2.waitKey()
cv2.imwrite('./hello_gray.jpg', gray_img)
