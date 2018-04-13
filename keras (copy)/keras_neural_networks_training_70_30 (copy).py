from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import numpy as np
#import random
import time

start_time = time.time()

X_train = np.empty((0,336))
Y_train = np.array([])
X_test = np.empty((0,336))
Y_test = np.array([])
#count = 0
train = [76, 8, 111, 75, 50, 74, 25, 56, 55, 83, 102, 129, 9, 17, 85, 101, 35, 117, 107, 64, 104, 97, 95, 38, 14, 96, 81, 93, 67, 48, 112, 62, 87, 115, 61, 109, 11, 119, 36, 4, 22, 30, 89, 66, 71, 33, 114, 60, 21, 37, 24, 100, 34, 121, 16, 10, 51, 5, 18, 6, 128, 77, 110, 125, 53, 43, 68, 103, 108, 65, 58, 19, 39, 2, 3, 28, 122, 127, 84, 59, 13, 90, 105, 54, 98, 29, 94, 124, 26, 1, 46, 42, 76, 8, 111, 75, 50, 74, 25, 56, 55, 83, 102, 129, 9, 17, 85, 101, 35, 117, 107, 64, 104, 97, 95, 38, 14, 96, 81, 93, 67, 48, 112, 62, 123, 87, 115, 61, 109, 11, 119, 36, 4, 22, 30, 89, 66, 71, 33, 114, 60, 21, 37, 24, 100, 34, 121, 16, 10, 51, 5, 18, 6, 128, 77, 110, 125, 53, 43, 68, 103, 108, 65, 58, 19, 39, 2, 3, 28, 122, 127, 84, 59, 13, 90, 105, 54, 98, 29, 94, 124, 26, 1, 46]

test = [70, 113, 12, 82, 78, 92, 23, 45, 80, 73, 120, 40, 27, 118, 69, 52, 88, 86, 32, 7, 126, 47, 41, 15, 57, 31, 63, 91, 49, 116, 72, 106, 20, 44, 99, 42, 70, 113, 12, 82, 78, 92, 23, 45, 80, 73, 120, 79, 40, 27, 118, 69, 52, 88, 86, 32, 7, 126, 47, 41, 15, 57, 31, 63, 91, 49, 116, 72, 106, 20, 44, 99]
#random.shuffle(data)

for k in range(0,len(train)):
	index = train[k]
	if k < 93:
		try:
        		file_name = 'violent_features_NON_VIOLENT/nonvio_' + str(index) + '.txt'
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
	else:
		try:
        		file_name = 'violent_features_VIOLENT/vio_' + str(index) + '.txt'
        		file_obj = open(file_name, 'r')
        		vif = np.loadtxt(file_obj)
        		if vif.shape[0] == 630:  # avoiding hd videos
            			continue
        		vif = np.reshape(vif, (-1, vif.shape[0]))
			X_train = np.vstack((X_train, vif))
			Y_train = np.append(Y_train,1)
			file_obj.close()
		except:
        		continue
        		print 'error in reading vio_%d.txt' % i

for k in range(0,len(test)):
	index = test[k]
	if k < 37:
		try:
        		file_name = 'violent_features_NON_VIOLENT/nonvio_' + str(index) + '.txt'
        		file_obj = open(file_name, 'r')
        		vif = np.loadtxt(file_obj)
       			if vif.shape[0] == 630:  # avoiding hd videos
            			continue
        		vif = np.reshape(vif, (-1, vif.shape[0]))
			X_test = np.vstack((X_test,vif))
			Y_test = np.append(Y_test,0)
			file_obj.close()
		except:
		        continue
        		print 'error in reading nonvio_%d.txt' % i
	else:
		try:
        		file_name = 'violent_features_VIOLENT/vio_' + str(index) + '.txt'
        		file_obj = open(file_name, 'r')
        		vif = np.loadtxt(file_obj)
        		if vif.shape[0] == 630:  # avoiding hd videos
            			continue
        		vif = np.reshape(vif, (-1, vif.shape[0]))
			X_test = np.vstack((X_test, vif))
			Y_test = np.append(Y_test,1)
			file_obj.close()
		except:
        		continue
        		print 'error in reading vio_%d.txt' % i


seed = 7
np.random.seed(seed)
model = Sequential()
model.add(Dense(350, activation="relu", kernel_initializer="uniform", input_dim=336))
model.add(Dense(336, activation='relu', kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=150, batch_size=2,  verbose=0)

predictions = model.predict(X_test)

pred = [round(x[0]) for x in predictions]


acc_count = 0
for k in range(0,len(pred)):
    if pred[k] == Y_test[k]:
        acc_count += 1

accuracy = float(acc_count)/len(pred)
print 'accuracy is : ' + str(accuracy)
print("--- %s seconds ---" % (time.time() - start_time))

cm = confusion_matrix(Y_test,pred)
print ""
print "confusion matrix"
print cm





