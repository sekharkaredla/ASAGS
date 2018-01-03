import cv2
from VideoProcess import PreProcess
from OpticalFlow import OptFlow

vid = PreProcess()
vid.read_video('vio_1.avi')
flow = OptFlow()
vid.setVideoDimension(100)

for each_frame_index in range(3,vid.total_frames - vid.FRAME_GAP - 5,vid.FRAME_GAP):
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

    print m1,m2
