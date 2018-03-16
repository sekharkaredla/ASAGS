# ffmpeg -i test.avi -vf scale=320:240 test1.avi
# to resize videos to 240 rows and 320 coloums
import cv2
import numpy as np
from VideoProcess import PreProcess
from OpticalFlow import OptFlow
import math
from keras.models import model_from_json

json_file = open('model_100.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_100.h5")

video_name = 'test.avi'

vid = PreProcess()
vid.read_video(video_name)
flow = OptFlow()
vid.setVideoDimension(100)
index = 0
height = 0
width = 0
B_height = 0
B_width = 0
bins = np.arange(0.0,1.05,0.05,dtype=np.float64)
temp_flows = []

def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape,dtype=np.float64)
    for i in map_to_bins:
        r[i-1] += 1
    return r

def getBlockHist(flow_video):
    flow_vec = np.reshape(flow_video,(flow_video.shape[0]*flow_video.shape[1],1))
    count_of_bins = histc(flow_vec,bins)
    return count_of_bins/np.sum(count_of_bins)

def getFrameHist(flow_video_size):
	global temp_flows,index
	flow_video = np.zeros(flow_video_size,dtype=np.float64)
	for each_flow in temp_flows:
	    flow_video = flow_video + each_flow
	flow_video = flow_video / index
	index = 0
	temp_flows = []
	height = flow_video.shape[0]
	width = flow_video.shape[1]
	B_height = int(math.floor((height - 11)/4))
	B_width = int(math.floor((width - 11)/4))
	frame_hist = []
	for y in range(6,height-B_height-5,B_height):
	    for x in range(6,width-B_width-5,B_width):
	        block_hist = getBlockHist(flow_video[y:y+B_height-1,x:x+B_width-1])
	        frame_hist = np.append(frame_hist,block_hist,axis = 0)
	return frame_hist


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


    change_mag = abs(m2-m1)
    binary_mag = np.ones(change_mag.shape,dtype=np.float64)
    threshold = np.mean(change_mag , dtype=np.float64)
    temp_flows.append(np.where(change_mag < threshold,0,binary_mag))

    if index > 9:
        vif = getFrameHist(CURRENT_F.shape)
        X_frame = np.empty((0,252))
        vif = np.reshape(vif, (-1, vif.shape[0]))
        X_frame = np.vstack((X_frame, vif))
        pred = model.predict(X_frame)
        print pred
