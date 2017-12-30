import numpy
import cv2
import bob.ip.optflow.liu.sor


def getFrameResized(cap):
    ret , frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rescale = 100.0/(frame.shape[1])
    if rescale<0.8:
        dim = (100, int(frame.shape[0] * rescale))
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame

def calculateOpticalFLow(frame1,frame2):
    alpha = 0.0026
    ratio = 0.6
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    (vx,vy,warpI2) = bob.ip.optflow.liu.sor.flow(frame1,frame2,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations)
    cv2.imshow('vx',vx)
    cv2.imshow('vy',vy)

def main():
    cap = cv2.VideoCapture(0)
    while True:
        frame1 = getFrameResized(cap)
        getFrameResized(cap)
        getFrameResized(cap)
        frame2 = getFrameResized(cap)
        calculateOpticalFLow(frame1,frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
