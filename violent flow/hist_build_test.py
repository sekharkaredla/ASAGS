from violent_flow_test import getViolentFlow
import math
import numpy as np


flow_video = getViolentFlow('vio_1.avi')

height = flow_video.shape[0]
width = flow_video.shape[1]

B_height = int(math.floor((height - 11)/4))
B_width = int(math.floor((width - 11)/4))

def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape,dtype=np.float64)
    for i in map_to_bins:
        r[i-1] += 1
    return r

bins = np.arange(0.0,1.05,0.05,dtype=np.float64)


def getBlockHist(flow_video):
    flow_vec = np.reshape(flow_video,(flow_video.shape[0]*flow_video.shape[1],1))
    count_of_bins = histc(flow_vec,bins)
    return count_of_bins/np.sum(count_of_bins)

def getFeatureVector():
    frame_hist = []
    for y in range(6,height-B_height-5,B_height):
        for x in range(6,width-B_width-5,B_width):
            block_hist = getBlockHist(flow_video[y:y+B_height-1,x:x+B_width-1])
            frame_hist = np.append(frame_hist,block_hist,axis = 0)
    return frame_hist


getFeatureVector()
