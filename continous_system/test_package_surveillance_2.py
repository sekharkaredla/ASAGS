import time
from ContSurv import ContinousSurv2
if __name__ == '__main__':
    start_time = time.time()
	obj = ContinousSurv2()
	obj.setVideoName('testV2.avi')
	obj.doSurveillanceFromVideo()
	# obj.doSurveillanceFromCamera()
    print '-----------------------------'
    print 'total time taken : ' + str(start_time - time.time())
