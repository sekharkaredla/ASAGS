#cleafrom keras.models import Sequential
#from keras.layers import Dense
import numpy as np

np.random.seed(7)

X_train = []
Y_train = []
X_test = []
Y_test = []
count = 0
data = range(1,130)
random.shuffle(data)

for i in data:
    try:
        file_name = 'violent_features_NON_VIOLENT/nonvio_' + str(i) + '.txt'
        file_obj = open(file_name, 'r')
        vif = np.loadtxt(file_obj)
        print vif.shape
        if vif.shape[0] == 630:  # avoiding hd videos
            continue
        if count < 92:
            X_train.append(vif)
            Y_train.append(0)
        else:
            X_test.append(vif)
            Y_test.append(0)
        file_obj.close()
        count += 1
    except:
        continue
        print 'error in reading nonvio_%d.txt' % i
# reading violent video features
count = 0
for i in data:
    try:
        file_name = 'violent_features_VIOLENT/vio_' + str(i) + '.txt'
        file_obj = open(file_name, 'r')
        vif = np.loadtxt(file_obj)
        if vif.shape[0] == 630:  # avoiding hd videos
            continue
        if count < 92:
            X_train.append(vif)
            Y_train.append(1)
        else:
            X_test.append(vif)
            Y_test.append(1)
        file_obj.close()
        count += 1
    except:
        continue
        print 'error in reading vio_%d.txt' % i
