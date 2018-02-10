import numpy as np
import cv2

def resize_frame(frame):
    rescale = 100.0/(frame.shape[1])
    if rescale<0.8:
        dim = (100, int(frame.shape[0] * rescale))
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame


cap = cv2.VideoCapture('vio_1.avi')
ret, frame = cap.read()
cv2.imwrite('image_before_resize.png',frame)
cv2.imwrite('image_after_resize.png',resize_frame(frame))

cap.release()
cv2.destroyAllWindows()
