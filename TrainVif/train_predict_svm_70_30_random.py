from sklearn import svm
import numpy as np
import time
import random
acc_all = 0.0
for i in range(1,21):
    start_time = time.time()
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    count = 0
    data = range(1,130)
    random.shuffle(data)
    #reading non violent video features
    for i in data:
        try:
            file_name = 'violent_features_NON_VIOLENT/nonvio_'+str(i)+'.txt'
            file_obj = open(file_name,'r')
            vif = np.loadtxt(file_obj)
            if vif.shape[0] == 336:
                continue
            if count < 92:
                X_train.append(vif)
                Y_train.append(0)
            else:
                X_test.append(vif)
                Y_test.append(0)
            file_obj.close()
            count+=1
        except:
            continue
            print 'error in reading nonvio_%d.txt'%i
    #reading violent video features
    count = 0
    for i in data:
        try:
            file_name = 'violent_features_VIOLENT/vio_'+str(i)+'.txt'
            file_obj = open(file_name,'r')
            vif = np.loadtxt(file_obj)
            if vif.shape[0] == 336:
                continue
            if count < 92:
                X_train.append(vif)
                Y_train.append(1)
            else:
                X_test.append(vif)
                Y_test.append(1)
            file_obj.close()
            count+=1
        except:
            continue
            print 'error in reading vio_%d.txt'%i

    #print len(X_train)
    #print len(X_test)
    #training
    clf = svm.SVC(kernel = 'linear')
    clf.fit(X_train,Y_train)
    print clf
    print("--- %s seconds ---" % (time.time() - start_time))


    #predicting
    pred = []

    for i in X_test:
        pred.append(clf.predict(i.reshape(1,-1)))

    count = 0

    for i in range(0,len(Y_test)):
        if pred[i][0] == Y_test[i]:
            count = count + 1

    accuracy = float(count)/len(Y_test)
    print 'accuracy is : ' + str(accuracy)
    acc_all = acc_all + accuracy
print 'overall : ' + str(acc_all/20.0)
