from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
import time

ovr_acc = 0.0
iters = 0
start_time = time.time()

data_violent = range(1,130)
data_nonviolent = range(1,130)
random.shuffle(data_violent)
random.shuffle(data_nonviolent)

for i in range(20,131,20):
    X_train = np.empty((0,336))
    Y_train = np.array([])
    X_test = np.empty((0,336))
    Y_test = np.array([])
    iters += 1
    test_set = range(i-20,i)
    for j in test_set:
        try:
            file_name = 'violent_features_NON_VIOLENT/nonvio_'+str(data_nonviolent[j])+'.txt'
            file_obj = open(file_name,'r')
            vif = np.loadtxt(file_obj)
            if vif.shape[0] == 630:# avoiding hd videos
                continue
            vif = np.reshape(vif,(-1,vif.shape[0]))
            X_test = np.vstack((X_test,vif))
            Y_test = np.append(Y_test,0)
            file_obj.close()
        except:
            continue
            print 'error in reading nonvio_%d.txt'%data_nonviolent[i]
        try:
            file_name = 'violent_features_VIOLENT/vio_'+str(data_violent[j])+'.txt'
            file_obj = open(file_name,'r')
            vif = np.loadtxt(file_obj)
            if vif.shape[0] == 630:# avoiding hd videos
                continue
            vif = np.reshape(vif,(-1,vif.shape[0]))
            X_test = np.vstack((X_test,vif))
            Y_test = np.append(Y_test,1)
            file_obj.close()
        except:
            continue
            print 'error in reading nonvio_%d.txt'%data_violent[i]
    for j in range(1,130):
        try:
            if j in test_set:
                continue
            file_name = 'violent_features_NON_VIOLENT/nonvio_'+str(data_nonviolent[j])+'.txt'
            file_obj = open(file_name,'r')
            vif = np.loadtxt(file_obj)
            if vif.shape[0] == 630:# avoiding hd videos
                continue
            vif = np.reshape(vif,(-1,vif.shape[0]))
            X_train = np.vstack((X_train,vif))
            Y_train = np.append(Y_train,0)
            file_obj.close()
        except:
            continue
            print 'error in reading nonvio_%d.txt'%j
    for j in range(1,130):
        try:
            if j in test_set:
                continue
            file_name = 'violent_features_VIOLENT/vio_'+str(data_violent[j])+'.txt'
            file_obj = open(file_name,'r')
            vif = np.loadtxt(file_obj)
            if vif.shape[0] == 630:# avoiding hd videos
                continue
            vif = np.reshape(vif,(-1,vif.shape[0]))
            X_train = np.vstack((X_train,vif))
            Y_train = np.append(Y_train,1)
            file_obj.close()
        except:
            continue
            print 'error in reading vio_%d.txt'%j

    if len(X_train) == 0:
        iters -= 1
        continue

    seed = 7
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(350, activation="relu", kernel_initializer="uniform", input_dim=336))

    for l in range(1,5):
        model.add(Dense(336, activation='relu', kernel_initializer="uniform"))

    model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=150, batch_size=10,  verbose=0)

    predictions = model.predict(X_test)

    pred = [round(x[0]) for x in predictions]


    acc_count = 0
    for k in range(0,len(pred)):
        if pred[k] == Y_test[k]:
            acc_count += 1

    accuracy = float(acc_count)/len(pred)
    print 'accuracy is : ' + str(accuracy)
    ovr_acc += accuracy
print 'average accuracy is : ' + str(ovr_acc/iters)
