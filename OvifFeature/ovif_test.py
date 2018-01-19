import time
from Ovif import OvifCalc
obj = OvifCalc('input1.avi')
start_time = time.time()
feature = obj.getOvifFeature()
print feature,feature.shape
print("--- %s seconds ---" % (time.time() - start_time))
