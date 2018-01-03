import cv2
from VideoProcess import PreProcess
from OpticalFlow import OptFlow

vid = PreProcess()
flow = OptFlow()
vid.useCamera()
vid.setVideoDimension(300)
while True:
	frame1 = vid.getFramesFromSource()
	frame2 = vid.getFramesFromSource()
	(vx,vy,w) = flow.sorFlow(frame1,frame2)

	cv2.imshow('camera',frame2)
	cv2.imshow('vy',vy)
	cv2.imshow('vx',vx)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
