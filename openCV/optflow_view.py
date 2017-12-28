import bob.ip.optflow.liu.sor
import numpy
import cv2
cap = cv2.VideoCapture('vio_1.avi')
ret1 , frame1 = cap.read()
ret2 , frame2 = cap.read()
frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
# frame1 = float(frame1)
# frame2 = float(frame2)
alpha = 0.0026
ratio = 0.6
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations]
while True:
	#cv2.imshow('temp1',frame1)
	#cv2.imshow('temp2',frame2)
	(vx,vy,warpI2) = bob.ip.optflow.liu.sor.flow(frame1,frame2)
	cv2.imshow('temp',vx)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break