from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import numpy as np
import random
import time

data = range(1,130)
random.shuffle(data)
X_train = np.empty((0,336))
Y_train = np.array([])
for i in data:
    try:
        file_name = 'violent_features_NON_VIOLENT/nonvio_' + str(i) + '.txt'
        file_obj = open(file_name, 'r')
        vif = np.loadtxt(file_obj)
        if vif.shape[0] == 630:  # avoiding hd videos
            continue
        vif = np.reshape(vif, (-1, vif.shape[0]))

        X_train = np.vstack((X_train,vif))
        Y_train = np.append(Y_train,0)

        file_obj.close()
    except:
        continue
        print 'error in reading nonvio_%d.txt' % i

# reading violent video features
for i in data:
    try:
        file_name = 'violent_features_VIOLENT/vio_' + str(i) + '.txt'
        file_obj = open(file_name, 'r')
        vif = np.loadtxt(file_obj)
        if vif.shape[0] == 630:  # avoiding hd videos
            continue
        vif = np.reshape(vif, (-1, vif.shape[0]))

        X_train = np.vstack((X_train, vif))
        Y_train = np.append(Y_train, 1)

        file_obj.close()
    except:
        continue
        print 'error in reading vio_%d.txt' % i

seed = 7
np.random.seed(seed)
model = Sequential()
model.add(Dense(350, activation="relu", kernel_initializer="uniform", input_dim=336))

for l in range(1,2):
    model.add(Dense(336, activation='relu', kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=150, batch_size=10,  verbose=0)

print 'model trained'

ovr_acc = 0.0
start_time = time.time()
for j in range(1,21):
    random.shuffle(data)
    X_test = np.empty((0,336))
    Y_test = np.array([])
    count = 0
    for i in data:
        try:
            file_name = 'violent_features_NON_VIOLENT/nonvio_' + str(i) + '.txt'
            file_obj = open(file_name, 'r')
            vif = np.loadtxt(file_obj)
            if vif.shape[0] == 630:  # avoiding hd videos
                continue
            vif = np.reshape(vif, (-1, vif.shape[0]))
            if count%2==0 and len(Y_test)<=39:
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
            if count%2==0 and len(Y_test)<=78:
                X_test = np.vstack((X_test, vif))
                Y_test = np.append(Y_test, 1)

            file_obj.close()
            count += 1
        except:
            continue
            print 'error in reading vio_%d.txt' % i




    predictions = model.predict(X_test)

    pred = [round(x[0]) for x in predictions]


    acc_count = 0
    for k in range(0,len(pred)):
        if pred[k] == Y_test[k]:
            acc_count += 1

    cm = confusion_matrix(Y_test, pred)
    print cm

    accuracy = float(acc_count)/len(pred)
    print 'accuracy is : ' + str(accuracy)
    ovr_acc += accuracy

print 'overall : ' + str(ovr_acc/20.0)
print("--- %s seconds ---" % (time.time() - start_time))

model_json = model.to_json()

with open("model_100.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_100.h5")
print("Saved model to disk")
