import cv2
import numpy as np

def nothing(x):
    pass

# Create a black image, a window
bgr = cv2.imread('test.jpeg')
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
img = bgr
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('low_h','image',0,255,nothing)
cv2.createTrackbar('high_h','image',0,255,nothing)
cv2.createTrackbar('low_s','image',0,255,nothing)
cv2.createTrackbar('high_s','image',0,255,nothing)
cv2.createTrackbar('low_v','image',0,255,nothing)
cv2.createTrackbar('high_v','image',0,255,nothing)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    low_h = cv2.getTrackbarPos('low_h','image')
    low_s = cv2.getTrackbarPos('low_s','image')
    low_v = cv2.getTrackbarPos('low_v','image')
    high_h = cv2.getTrackbarPos('high_h','image')
    high_s = cv2.getTrackbarPos('high_s','image')
    high_v = cv2.getTrackbarPos('high_v','image')

    lower_bound = np.array([low_h, low_s, low_v])
    high_bound = np.array([high_h, high_s, high_v])

    mask = cv2.inRange(hsv, lower_bound, high_bound)
    img = cv2.bitwise_and(bgr, bgr, mask=mask)

cv2.destroyAllWindows()
