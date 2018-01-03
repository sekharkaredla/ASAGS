import numpy as np
from VideoProcess import PreProcess
from OpticalFlow import OptFlow

vid = PreProcess()
flow = OptFlow()

vid.read_video('vio_1.avi')
vid.setVideoDimension(200)

frame1 = vid.getFramesFromSource()
frame2 = vid.getFramesFromSource()
frame2 = vid.getFramesFromSource()
frame2 = vid.getFramesFromSource()

(vx,vy,w) = flow.sorFlow(frame1,frame2)

flow_magnitude = np.sqrt(np.square(vx) + np.square(vy))

print flow_magnitude
