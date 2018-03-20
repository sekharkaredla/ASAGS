import cv2
import numpy as np
from VideoProcess import PreProcess
from OpticalFlow import OptFlow
import math
from keras.models import model_from_json
import time

class ContinousSurv:
    def __init__(self):
        json_file = open('model_100.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("model_100.h5")
        print 'loaded model from disk'

        self.vid = PreProcess()
        self.vid.setVideoDimension(100)
        self.flow = OptFlow()
        self.height = 0
        self.width = 0
        self.B_height = 0
        self.B_width = 0
        self.index = 0
        self.temp_flows = []
        self.bins = np.arange(0.0,1.05,0.05,dtype=np.float64)

    def setVideoName(self,video_name):
        self.vid.read_video(video_name)

    def histc(self,X, bins):
        map_to_bins = np.digitize(X,bins)
        r = np.zeros(bins.shape,dtype=np.float64)
        for i in map_to_bins:
            r[i-1] += 1
        return r

    def getBlockHist(self,flow_video):
        flow_vec = np.reshape(flow_video,(flow_video.shape[0]*flow_video.shape[1],1))
        count_of_bins = self.histc(flow_vec,self.bins)
        return count_of_bins/np.sum(count_of_bins)

    def getFrameHist(self,flow_video_size):
    	flow_video = np.zeros(flow_video_size,dtype=np.float64)
    	for each_flow in self.temp_flows:
    	    flow_video = flow_video + each_flow
    	flow_video = flow_video / self.index
    	self.index = 0
    	self.temp_flows = []
    	self.height = flow_video.shape[0]
    	self.width = flow_video.shape[1]
    	self.B_height = int(math.floor((self.height - 11)/4))
    	self.B_width = int(math.floor((self.width - 11)/4))
    	frame_hist = []
    	for y in range(6,self.height-self.B_height-4,self.B_height):
    	    for x in range(6,self.width-self.B_width-4,self.B_width):
    	        block_hist = self.getBlockHist(flow_video[y:y+self.B_height,x:x+self.B_width])
    	        frame_hist = np.append(frame_hist,block_hist,axis = 0)
    	return frame_hist

    def doSurveillanceFromVideo(self):
        FPS = round(self.vid.getFPS())
        print 'FPS is : '+str(FPS)
        while True:
            frames = self.vid.getFramesFromVideoSource()
            PREV_F = frames[0]
            CURRENT_F = frames[1]
            NEXT_F = frames[2]

            frame_number = frames[3]

            PREV_F = self.vid.resize_frame(PREV_F)
            CURRENT_F = self.vid.resize_frame(CURRENT_F)
            NEXT_F = self.vid.resize_frame(NEXT_F)

            (vx1,vy1,w1) = self.flow.sorFlow(PREV_F,CURRENT_F)
            (vx2,vy2,w2) = self.flow.sorFlow(CURRENT_F,NEXT_F)

            m1 = self.flow.getFlowMagnitude(vx1,vy1)
            self.index = self.index + 1
            m2 = self.flow.getFlowMagnitude(vx2,vy2)

            change_mag = abs(m2-m1)
            binary_mag = np.ones(change_mag.shape,dtype=np.float64)
            threshold = np.mean(change_mag , dtype=np.float64)
            self.temp_flows.append(np.where(change_mag < threshold,0,binary_mag))

            if self.index>=int(FPS/3):
                vif = self.getFrameHist(CURRENT_F.shape)
                X_frame = np.empty((0,336))
                vif = np.reshape(vif, (-1, vif.shape[0]))
                X_frame = np.vstack((X_frame, vif))
                pred = self.model.predict(X_frame)
                pred = round(pred[0][0])
                if pred == 1:
                    time_violence = float(frame_number) / self.vid.fps
                    print 'violent  ---   '+str(int(time_violence))+' seconds'

    def doSurveillanceFromCamera(self):
        start_time = time.time()
        self.vid.useCamera()
        FPS = round(self.vid.getFPS())
        print 'FPS is :'+str(FPS)
        while True:
            frames = self.vid.getFramesFromCameraSource()
            PREV_F = frames[0]
            CURRENT_F = frames[1]
            NEXT_F = frames[2]

            time_now = frames[3]

            PREV_F = self.vid.resize_frame(PREV_F)
            CURRENT_F = self.vid.resize_frame(CURRENT_F)
            NEXT_F = self.vid.resize_frame(NEXT_F)

            (vx1,vy1,w1) = self.flow.sorFlow(PREV_F,CURRENT_F)
            (vx2,vy2,w2) = self.flow.sorFlow(CURRENT_F,NEXT_F)

            m1 = self.flow.getFlowMagnitude(vx1,vy1)
            self.index = self.index + 1
            m2 = self.flow.getFlowMagnitude(vx2,vy2)

            change_mag = abs(m2-m1)
            binary_mag = np.ones(change_mag.shape,dtype=np.float64)
            threshold = np.mean(change_mag , dtype=np.float64)
            self.temp_flows.append(np.where(change_mag < threshold,0,binary_mag))

            if self.index>=int(FPS/3):
                vif = self.getFrameHist(CURRENT_F.shape)
                X_frame = np.empty((0,336))
                vif = np.reshape(vif, (-1, vif.shape[0]))
                X_frame = np.vstack((X_frame, vif))
                pred = self.model.predict(X_frame)
                print pred,time_now-start_time
                pred = round(pred[0][0])
                if pred == 1:
                    print 'violent  ---   '+str(time_now - start_time)
