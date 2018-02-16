import bob.ip.optflow.liu.sor
import numpy
import cv2

def getFrameResized(frame):

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    size = 100
    rescale = float(size)/(frame.shape[1])
    if rescale<0.8:
        dim = (size, int(frame.shape[0] * rescale))
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame

def calculateOpticalFLow(frame1,frame2,filename):
    #dont use parameters for visualizations
    alpha = 0.0026
    ratio = 0.6
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    temp = ()
    temp = bob.ip.optflow.liu.sor.flow(frame1,frame2,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations)
    #(vx,vy,warpI2) = bob.ip.optflow.liu.sor.flow(frame1,frame2)
    if temp == ():
        print 'ERROR IN '+filename

def main(path,filename):
    cap = cv2.VideoCapture(path)
    while(cap.isOpened()):
        ret , frame1 = cap.read()
        cap.read()
        cap.read()
        ret , frame2 = cap.read()
        frame1 = getFrameResized(frame1)
        frame2 = getFrameResized(frame2)
        calculateOpticalFLow(frame1,frame2,filename)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    file_vio = open('violent_list.txt')
    path = '/Users/roshni/Desktop/VideoData/Violence/'
    for each_file in file_vio.readlines():
        each_file = each_file[:-1]
        print each_file + '-----------------------------------------------------'
        try:
            main(path + each_file,each_file)
            print each_file + '  done'
        except:
            print 'error in  ' + each_file
