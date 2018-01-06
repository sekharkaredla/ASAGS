from violent_flow_test import getViolentFlow
import math
import numpy as np


flow_video = getViolentFlow('vio_1.avi')

height = flow_video.shape[0]
width = flow_video.shape[1]

B_height = math.floor((height - 11)/4)
B_width = math.floor((width - 11)/4)

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

print getBlockHist(flow_video)