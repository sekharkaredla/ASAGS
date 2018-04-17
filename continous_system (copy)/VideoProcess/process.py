import numpy
import cv2

class PreProcess:
    def __init__(self):
        #constants----------------
        self.FRAME_RATE = 25 #25 frames per second
        self.MOVEMENT_INTERVAL = 3 #difference between considered frames
        self.N = 4 #number of vertical blocks per frame
        self.M = 4 #number of horizontal blocks per frame
        self.FRAME_GAP = 2 * self.MOVEMENT_INTERVAL
        #-------------------------
        self.cap = ''
        self.total_frames = 0
        self.fps = 0
        self.time = 0
        #-------------------------
        self.dim = 100

    def read_video(self,video_name):
        self.cap = cv2.VideoCapture(video_name)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.time = self.total_frames / self.fps

    def getFrameFromIndex(self,frame_no):
        #Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
        #The second argument defines the frame number in range 0.0-1.0
        self.cap.set(1,frame_no)
        ret , img = self.cap.read()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img

    def resize_frame(self,frame):
        rescale = float(self.dim)/(frame.shape[1])
        if rescale<0.8:
            dim = (self.dim, int(frame.shape[0] * rescale))
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        return frame

    def setVideoDimension(self,dim):
        self.dim = dim

    def useCamera(self):
        self.cap = cv2.VideoCapture(0)

    def showInputFromCamera(self):
        while True:
            ret , frame = self.cap.read()
            cv2.imshow('camera',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def getFramesFromSource(self):
        ret , frame = self.cap.read()
        frame = self.resize_frame(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        return frame
