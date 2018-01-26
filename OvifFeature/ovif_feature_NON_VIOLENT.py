
from Ovif import OvifCalc
file_vio = open('non_violent_list.txt')
path = '/Users/roshni/Desktop/VideoData/Non-Violence/'
for each_file in file_vio.readlines():
    try:
        each_file = each_file[:-1]
        feature = OvifCalc(path + each_file)
        out_file = each_file[:-3] + 'txt'
        print '-----------------------------------------------------'
        print each_file
        feature.writeFeatureToFile('Ovif_features_NON_VIOLENT/' + out_file)
        print each_file + '  done'
    except Exception as err:
        print err
