from ViolentFlow import VioFlow
file_vio = open('nonviolent_list.txt')
path = '/home/dasarada/Desktop/Non-Violence/'
for each_file in file_vio.readlines():
    each_file = each_file[:-1]
    feature = VioFlow(path + each_file)
    out_file = each_file[:-3] + 'txt'
    try:
        feature.writeFeatureToFile('nonviolent_features_of_all_videos/' + out_file)
        print each_file + '  done'
    except:
        print 'error in  ' + each_file
