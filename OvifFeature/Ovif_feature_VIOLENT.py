import time
from Ovif import OvifCalc
file_vio = open('violent_list.txt')
path = '/home/dasarada/Desktop/Violence/'
start_time = time.time()
for each_file in file_vio.readlines():
    each_file = each_file[:-1]
    feature = OvifCalc(path + each_file)
    out_file = each_file[:-3] + 'txt'
    print each_file + '-----------------------------------------------------'
    try:
        feature.writeFeatureToFile('Ovif_features_VIOLENT/' + out_file)
        print each_file + '  done'
    except:
        print 'error in  ' + each_file
print("--- %s seconds ---" % (time.time() - start_time))
