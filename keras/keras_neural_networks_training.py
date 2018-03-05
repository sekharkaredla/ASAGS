from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random


X_train = np.empty((0,252))
Y_train = np.array([])
X_test = np.empty((0,252))
Y_test = np.array([])
count = 0
data = range(1,130)
random.shuffle(data)

for i in data:
    try:
        file_name = 'violent_features_NON_VIOLENT/nonvio_' + str(i) + '.txt'
        file_obj = open(file_name, 'r')
        vif = np.loadtxt(file_obj)
        if vif.shape[0] == 630:  # avoiding hd videos
            continue
        vif = np.reshape(vif, (-1, vif.shape[0]))
        if count < 92:
            X_train = np.vstack((X_train,vif))
            Y_train = np.append(Y_train,0)
        else:
            X_test = np.vstack((X_test,vif))
            Y_test = np.append(Y_test,0)
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
        vif = np.reshape(vif, (-1, vif.shape[0]))
        if count < 92:
            X_train = np.vstack((X_train, vif))
            Y_train = np.append(Y_train, 1)
        else:
            X_test = np.vstack((X_test, vif))
            Y_test = np.append(Y_test, 1)
        file_obj.close()
        count += 1
    except:
        continue
        print 'error in reading vio_%d.txt' % i


