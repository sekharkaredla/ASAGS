import bob.ip.optflow.liu.sor
import numpy
import cv2
def resize_frame(frame):
    rescale = 700.0/(frame.shape[1])
    if rescale<0.8:
        dim = (500, int(frame.shape[0] * rescale))
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame
#dont use parameters for visualizations
alpha = 0.0026
ratio = 0.6
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
frame1 = cv2.imread('car1.jpg')
frame2 = cv2.imread('car2.jpg')
frame1 = resize_frame(cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY))
frame2 = resize_frame(cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY))
#(u,v,w) = bob.ip.optflow.liu.sor.flow(frame1,frame2,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations)
(u,v,w) = bob.ip.optflow.liu.sor.flow(frame1,frame2)
print u
print v
cv2.imwrite('car_opt_flow.jpg',u)
while True:
	cv2.imshow('frame1',frame1)
	cv2.imshow('frame2',frame2)
	cv2.imshow('opt_flow',u)

	if cv2.waitKey(1) & 0xFF == ord('q'):
            break
