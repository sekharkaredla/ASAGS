import cv2
import numpy as np
from VideoProcess import PreProcess
from OpticalFlow import OptFlow
import math

vid = PreProcess()
vid.read_video('vio_1.avi')
flow = OptFlow()
vid.setVideoDimension(100)
index = 0
all_hists1 = []
all_hists2 = []

def binData(X,bins,mag):
    hist = [0.0]*9
    inds = np.digitize(X,bins)
    for i in range(X.size):
        hist[inds[i][0]-1] = hist[inds[i][0]-1] + mag[i][0]
    return hist



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

    phi1 = np.arctan(vy1/vx1)
    phi2 = np.arctan(vy2/vx2)

    data_block_size = (m1.shape[0]*m1.shape[1])/(4*4)

    m1 = np.reshape(m1,(m1.shape[0]*m1.shape[1],1))
    m2 = np.reshape(m2,(m2.shape[0]*m2.shape[1],1))
    phi1 = np.reshape(phi1,(phi1.shape[0]*phi1.shape[1],1))
    phi2 = np.reshape(phi2,(phi2.shape[0]*phi2.shape[1],1))

    hist_bins = np.arange(9.)*(2*np.pi)/9
    hist_bins = np.degrees(hist_bins)
    phi1 = np.rad2deg(phi1)
    phi2 = np.rad2deg(phi2)
    phi1 = np.where(phi1 < 0,360 + phi1,phi1)
    phi2 = np.where(phi2 < 0,360 + phi2,phi2)

    for k in range(0,(4*4)):
        data_to_be_binned1 = []
        data_to_be_binned2 = []
        if k != (4*4)-1:
            data_to_be_binned1 = m1[data_block_size*k:data_block_size*(k+1)]
            temp_phi1 = phi1[data_block_size*k:data_block_size*(k+1)]
            data_to_be_binned2 = m2[data_block_size*k:data_block_size*(k+1)]
            temp_phi2 = phi2[data_block_size*k:data_block_size*(k+1)]
        else:
            data_to_be_binned1 = m1[data_block_size*k:]
            temp_phi1 = phi1[data_block_size*k:]
            data_to_be_binned2 = m2[data_block_size*k:]
            temp_phi2 = phi2[data_block_size*k:]

        hist1 = binData(temp_phi1,hist_bins,data_to_be_binned1)
        hist2 = binData(temp_phi2,hist_bins,data_to_be_binned2)

        all_hists1 = all_hists1 + hist1
        all_hists2 = all_hists2 + hist2

    # print len(all_hists1) , len(all_hists2)
print all_hists1 , all_hists2
