from sklearn import svm
import numpy as np
import random

data_violent = range(1,130)
data_nonviolent = range(1,130)

def strong_shuffle(numbers):
	random.shuffle(numbers)
	random.shuffle(numbers)
	random.shuffle(numbers)

strong_shuffle(data_violent)
strong_shuffle(data_nonviolent)

X_train = []
Y_train = []

X_test = []
Y_test = []

for i in data_nonviolent[0:80]:
    try:
        file_name = 'violent_features_NON_VIOLENT/nonvio_'+str(i)+'.txt'
        file_obj = open(file_name,'r')
        vif = np.loadtxt(file_obj)
        if vif.shape[0] == 336:# avoiding hd videos
            continue
        X_train.append(vif)
        Y_train.append(0)
        file_obj.close()
    except:
        continue
        print 'error in reading nonvio_%d.txt'%i
for i in data_violent[0:80]:
    try:
        file_name = 'violent_features_VIOLENT/vio_'+str(i)+'.txt'
        file_obj = open(file_name,'r')
        vif = np.loadtxt(file_obj)
        if vif.shape[0] == 336:# avoiding hd videos
            continue
        X_train.append(vif)
        Y_train.append(1)
        file_obj.close()
    except:
        continue
        print 'error in reading nonvio_%d.txt'%i
for i in data_nonviolent[81:]:
    try:
        file_name = 'violent_features_NON_VIOLENT/nonvio_'+str(i)+'.txt'
        file_obj = open(file_name,'r')
        vif = np.loadtxt(file_obj)
        if vif.shape[0] == 336:# avoiding hd videos
            continue
        X_test.append(vif)
        Y_test.append(0)
        file_obj.close()
    except:
        continue
        print 'error in reading nonvio_%d.txt'%i
for i in data_violent[81:]:
    try:
        file_name = 'violent_features_VIOLENT/vio_'+str(i)+'.txt'
        file_obj = open(file_name,'r')
        vif = np.loadtxt(file_obj)
        if vif.shape[0] == 336:# avoiding hd videos
            continue
        X_test.append(vif)
        Y_test.append(1)
        file_obj.close()
    except:
        continue
        print 'error in reading nonvio_%d.txt'%i

# train contains the training set and test contains the testing set

classifiers = {}
w = [1.0/(2*129)]*252
for each_feature in range(0,252):
    # generating classifier for each feature
    train_data_X = []
    train_data_Y = []
    i = 0
    for each_set in X_train:
        train_data_X.append(each_set[each_feature])
        train_data_Y.append(Y_train[i])
        i += 1
    clf = svm.SVC(kernel = 'linear')
    clf.fit(np.array(train_data_X).reshape(-1,1),np.array(train_data_Y).reshape(-1,1))
    classifiers[each_feature] = clf

#calculate error for each classifier
errors = [0.0]*252

for each_feature in range(0,252):
    
    
