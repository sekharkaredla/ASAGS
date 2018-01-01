import bob.ip.optflow.liu.sor
import numpy
import cv2

#constants----------------
FRAME_RATE = 25 #25 frames per second
MOVEMENT_INTERVAL = 3 #difference between considered frames
N = 4 #number of vertical blocks per frame
M = 4 #number of horizontal blocks per frame

K = 4
FRAME_GAP = 2 * MOVEMENT_INTERVAL
#-------------------------
#read video file
cap = cv2.VideoCapture('vio_1.avi')
#procedure for extracting frames through index in a video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
time = total_frames / fps
# print fps
# print time
def getFrameFromIndex(frame_no):
    #Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
    #The second argument defines the frame number in range 0.0-1.0
    cap.set(1,frame_no)
    ret , img = cap.read()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def resize_frame(frame):
    rescale = 100.0/(frame.shape[1])
    if rescale<0.8:
        dim = (100, int(frame.shape[0] * rescale))
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame
#dont use parameters for visualizations
alpha = 0.0026
ratio = 0.6
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30

#getFrameFromIndex(50)
for each_frame_index in range(3,total_frames - FRAME_GAP - 5,FRAME_GAP):
    PREV_F = getFrameFromIndex(each_frame_index)
    CURRENT_F = getFrameFromIndex(each_frame_index + MOVEMENT_INTERVAL)
    NEXT_F = getFrameFromIndex(each_frame_index + (2 * MOVEMENT_INTERVAL))


    PREV_F = resize_frame(PREV_F)
    CURRENT_F = resize_frame(CURRENT_F)
    NEXT_F = resize_frame(NEXT_F)

    # print PREV_F,CURRENT_F,NEXT_F

    #(vx1,vy1,w1) = bob.ip.optflow.liu.sor.flow(PREV_F,CURRENT_F,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations)
    #(vx2,vy2,w2) = bob.ip.optflow.liu.sor.flow(CURRENT_F,NEXT_F,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations)
    (vx1,vy1,w1) = bob.ip.optflow.liu.sor.flow(PREV_F,CURRENT_F)
    (vx2,vy2,w2) = bob.ip.optflow.liu.sor.flow(CURRENT_F,NEXT_F)
    cv2.imwrite('testVideo_'+str(each_frame_index)+'vx1.jpg',vx1)
    cv2.imwrite('testVideo_'+str(each_frame_index)+'vx2.jpg',vx2)
    cv2.imwrite('testVideo_'+str(each_frame_index)+'vy1.jpg',vy1)
    cv2.imwrite('testVideo_'+str(each_frame_index)+'vy2.jpg',vy2)
    while True:
        cv2.imshow(str(each_frame_index)+'vx1',vx1)
        cv2.imshow(str(each_frame_index)+'vx2',vx2)
        cv2.imshow(str(each_frame_index)+'vy1',vy1)
        cv2.imshow(str(each_frame_index)+'vy2',vy2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
