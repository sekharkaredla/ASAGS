from ViolentFlow import VioFlow
file_vio = open('violent_file_list.txt')
content = file_vio.read()
ls = content.split('\n')
path = '/home/dasarada/Desktop/Violence/'
for k in ls:
    feature = VioFlow(path + k)
    k = k[:-3] + 'txt'
    feature.writeFeatureToFile('feature_violent_videos/' + k)
