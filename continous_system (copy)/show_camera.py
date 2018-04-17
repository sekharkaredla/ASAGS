import cv2
import numpy

cap = cv2.VideoCapture(0)

# 0 may be the system web-cam
# 1 may be the external video camera detected

def resize_frame(frame):
    rescale = float(100)/(frame.shape[1])
    if rescale<0.8:
        dim = (100, int(frame.shape[0] * rescale))
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame

while True:
    ref , frame = cap.read()
    #ref returns true or false, whether frame has been captured or not
    #frame is the frame

    gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    #cvtColor is convert Color , Blue Green Red to Gray
    cv2.imshow('frame',frame)
    cv2.imshow('gray',resize_frame(gray_frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
