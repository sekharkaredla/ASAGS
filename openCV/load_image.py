import cv2
import numpy
#import matplotlib.pyplot as plt

image = cv2.imread('me.jpeg',1)
#cv2.IMREAD_GRAYSCALE = 0
#cv2.IMREAD_COLOR = 1
#cv2.IMREAD_UNCHANGED = -1

cv2.imshow('image2',image)

cv2.waitKey(0)
cv2.destroyAllWindows()
