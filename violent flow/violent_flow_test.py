import cv2
import numpy as np
from VideoProcess import PreProcess
from OpticalFlow import OptFlow

vid = PreProcess()
vid.read_video('vio_1.avi')
flow = OptFlow()
vid.setVideoDimension(100)

index = 0
for each_frame_index in range(3,vid.total_frames - vid.FRAME_GAP - 5,vid.FRAME_GAP):
    index = index + 1

    PREV_F = vid.getFrameFromIndex(each_frame_index)
    CURRENT_F = vid.getFrameFromIndex(each_frame_index + vid.MOVEMENT_INTERVAL)
    NEXT_F = vid.getFrameFromIndex(each_frame_index + (2 * vid.MOVEMENT_INTERVAL))

    PREV_F = vid.resize_frame(PREV_F)
    CURRENT_F = vid.resize_frame(CURRENT_F)
    NEXT_F = vid.resize_frame(NEXT_F)

    (vx1,vy1,w1) = flow.sorFlow(PREV_F,CURRENT_F)
    (vx2,vy2,w2) = flow.sorFlow(CURRENT_F,NEXT_F)

    m1 = flow.getFlowMagnitude(vx1,vy1)
    m2 = flow.getFlowMagnitude(vx2,vy2)

#    print m1
#    print m2

#the below part is different in tal hassner's implementation

    change_mag = abs(m2-m1)
    binary_mag = np.ones(change_mag.shape,dtype=np.float64)
    threshold = np.mean(change_mag , dtype=np.float64)
    binary_mag = np.where(change_mag <=  threshold,0,binary_mag)
    print threshold,binary_mag
