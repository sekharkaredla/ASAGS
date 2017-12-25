import cv2
import numpy

cap = cv2.VideoCapture(0)

# 0 may be the system web-cam
# 1 may be the external video camera detected

while True:
    ref , frame = cap.read()
    #ref returns true or false, whether frame has been captured or not
    #frame is the frame

    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    #cvtColor is convert Color , Blue Green Red to Gray
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()