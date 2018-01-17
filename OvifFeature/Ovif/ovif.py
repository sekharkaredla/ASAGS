import cv2
import numpy as np
from VideoProcess import PreProcess
from OpticalFlow import OptFlow
import math

class OvifCalc:
    def __init__(self,video_name):
        self.vid = PreProcess()
        self.vid.read_video(video_name)
        self.flow = OptFlow()
        self.vid.setVideoDimension(100)
        self.index = 0
        self.binary_mags = []

    def binData(self,X,bins,mag):
        hist = [0.0]*9
        inds = np.digitize(X,bins)
        for i in range(X.size):
            hist[inds[i][0]-1] = hist[inds[i][0]-1] + mag[i][0]
        return hist

    def getBinaryMag(self,hist1,hist2):
        hist1 = np.array(hist1)
        hist2 = np.array(hist2)
        change_mag = abs(hist1 - hist2)
        threshold = np.mean(change_mag , dtype=np.float64)
        binary_mag = np.ones(change_mag.shape,dtype=np.float64)
        binary_mag = np.where(change_mag < threshold,0,binary_mag)
        return binary_mag


    def getOvifFeature(self):
        for each_frame_index in range(3,self.vid.total_frames - self.vid.FRAME_GAP - 5,self.vid.FRAME_GAP):

            PREV_F = self.vid.getFrameFromIndex(each_frame_index)
            CURRENT_F = self.vid.getFrameFromIndex(each_frame_index + self.vid.MOVEMENT_INTERVAL)
            NEXT_F = self.vid.getFrameFromIndex(each_frame_index + (2 * self.vid.MOVEMENT_INTERVAL))

            PREV_F = self.vid.resize_frame(PREV_F)
            CURRENT_F = self.vid.resize_frame(CURRENT_F)
            NEXT_F = self.vid.resize_frame(NEXT_F)

            (vx1,vy1,w1) = self.flow.sorFlow(PREV_F,CURRENT_F)
            (vx2,vy2,w2) = self.flow.sorFlow(CURRENT_F,NEXT_F)

            m1 = self.flow.getFlowMagnitude(vx1,vy1)
            self.index = self.index + 1
            m2 = self.flow.getFlowMagnitude(vx2,vy2)

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

            all_hists1 = []
            all_hists2 = []
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

                hist1 = self.binData(temp_phi1,hist_bins,data_to_be_binned1)
                hist2 = self.binData(temp_phi2,hist_bins,data_to_be_binned2)

                all_hists1 = all_hists1 + hist1
                all_hists2 = all_hists2 + hist2

            binary_mag = self.getBinaryMag(all_hists1,all_hists2)
            self.binary_mags.append(binary_mag)
        return self.getAverageBinary()

    def getAverageBinary(self):
        ovif_video = np.zeros(self.binary_mags[0].shape,dtype=np.float64)
        for each_ovif in self.binary_mags:
            ovif_video = ovif_video + each_ovif

        return ovif_video / self.index
