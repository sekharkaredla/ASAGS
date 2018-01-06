import cv2
import numpy as np
from VideoProcess import PreProcess
from OpticalFlow import OptFlow

def getViolentFlow(video_name):
    vid = PreProcess()
    vid.read_video(video_name)
    flow = OptFlow()
    vid.setVideoDimension(100)
    index = 0
    temp_flows = []
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
        index = index + 1
        m2 = flow.getFlowMagnitude(vx2,vy2)

    #    print m1
    #    print m2

        change_mag = abs(m2-m1)
        binary_mag = np.ones(change_mag.shape,dtype=np.float64)
        threshold = np.mean(change_mag , dtype=np.float64)
        temp_flows.append(np.where(change_mag < threshold,0,binary_mag))

    flow_video = np.zeros(change_mag.shape,dtype=np.float64)
    for each_flow in temp_flows:
        flow_video = flow_video + each_flow

    flow_video = flow_video / index
    return flow_video

if __name__ == '__main__':
    print getViolentFlow('vio_1.avi')
