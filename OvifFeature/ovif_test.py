from Ovif import OvifCalc
obj = OvifCalc('vio_1.avi')
feature = obj.getOvifFeature()
print feature,feature.shape
