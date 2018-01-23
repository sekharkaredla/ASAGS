import time
from Ovif import OvifCalc
obj = OvifCalc('/home/dasarada/Desktop/Violence/vio_1.avi')
start_time = time.time()
feature = obj.getOvifFeature()
print feature,feature.shape
obj.writeFeatureToFile('test.txt')
print("--- %s seconds ---" % (time.time() - start_time))
