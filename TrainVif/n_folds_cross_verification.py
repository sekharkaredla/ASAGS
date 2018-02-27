from sklearn import svm
import numpy as np
import random

total_accuracy = 0.0
iters = 0

data_violent = range(1,130)
data_nonviolent = range(1,130)
random.shuffle(data_violent)
random.shuffle(data_nonviolent)

for i in range(10,131,10):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    iters += 1
    test_set = range(i-10,i)
    for j in test_set:
        try:
            file_name = 'violent_features_NON_VIOLENT/nonvio_'+str(data_nonviolent[j])+'.txt'
            file_obj = open(file_name,'r')
            vif = np.loadtxt(file_obj)
            if vif.shape[0] == 336:# avoiding hd videos
                continue
            X_test.append(vif)
            Y_test.append(0)
            file_obj.close()
        except:
            continue
            print 'error in reading nonvio_%d.txt'%data_nonviolent[i]
        try:
            file_name = 'violent_features_VIOLENT/vio_'+str(data_violent[j])+'.txt'
            file_obj = open(file_name,'r')
            vif = np.loadtxt(file_obj)
            if vif.shape[0] == 336:# avoiding hd videos
                continue
            X_test.append(vif)
            Y_test.append(1)
            file_obj.close()
        except:
            continue
            print 'error in reading nonvio_%d.txt'%data_violent[i]
    for j in range(1,130):
        try:
            if j in [data_nonviolent[l] for l in test_set]:
                continue
            file_name = 'violent_features_NON_VIOLENT/nonvio_'+str(j)+'.txt'
            file_obj = open(file_name,'r')
            vif = np.loadtxt(file_obj)
            if vif.shape[0] == 336:# avoiding hd videos
                continue
            X_train.append(vif)
            Y_train.append(0)
            file_obj.close()
        except:
            continue
            print 'error in reading nonvio_%d.txt'%j
    for j in range(1,130):
        try:
            if j in [data_violent[l] for l in test_set]:
                continue
            file_name = 'violent_features_VIOLENT/vio_'+str(j)+'.txt'
            file_obj = open(file_name,'r')
            vif = np.loadtxt(file_obj)
            if vif.shape[0] == 336:# avoiding hd videos
                continue
            X_train.append(vif)
            Y_train.append(1)
            file_obj.close()
        except:
            continue
            print 'error in reading vio_%d.txt'%j

    clf = svm.SVC(kernel = 'linear')
    if len(X_train) == 0:
        iters -= 1
        continue
    clf.fit(X_train,Y_train)
    print clf

    pred = []

    for i in X_test:
        pred.append(clf.predict(i.reshape(1,-1)))

    count = 0

    for i in range(0,len(Y_test)):
        if pred[i][0] == Y_test[i]:
            count = count + 1

    total_accuracy += float(count)/len(Y_test)
    print 'accuracy is : '+str(float(count)/len(Y_test))

print 'average accuracy is : ' + str(total_accuracy/iters)
