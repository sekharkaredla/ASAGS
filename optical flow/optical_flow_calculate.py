import bob.ip.optflow.liu.sor
import numpy
import cv2
cap = cv2.VideoCapture('vio_1.avi')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
time = total_frames / fps
# print fps
# print time
def getFrameFromIndex(frame_no):
    frame_seq = frame_no/total_frames
    #Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
    #The second argument defines the frame number in range 0.0-1.0
    cap.set(2,frame_seq);
    ret , img = cap.read()
    while True:
        cv2.imshow('temp',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

getFrameFromIndex(50)
