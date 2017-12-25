import numpy
import cv2

cap = cv2.VideoCapture('vio_1.avi')
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('temp',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()