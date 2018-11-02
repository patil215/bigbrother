import cv2

img = cv2.imread('./hello.jpg')
print("Image shape: {}".format(img.shape))
while True:
    bbox = cv2.selectROI("select", img, False)
    print(bbox)
