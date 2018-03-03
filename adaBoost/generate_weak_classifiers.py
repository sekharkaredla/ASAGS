# generating 20 weak classifiers 
# 50 for training
# 80 for testing(validation)

from sklearn import svm
import numpy as np
import random

def strong_shuffle(numbers):
	random.shuffle(numbers)
	random.shuffle(numbers)
	random.shuffle(numbers)

violence_indexes = range(1,130)
nonviolence_indexes = range(1,130)

X_test = []
Y_test = []

class Classifier:
	def __init__(self):
		strong_shuffle(violence_indexes)
		strong_shuffle(nonviolence_indexes)
		self.violence_indexes = violence_indexes[0:25]
		self.nonviolence_indexes = nonviolence_indexes[0:25]
		self.X_train = []
		self.Y_train = []

		for i in self.violence_indexes:
			try:
				file_name = 'violent_features_VIOLENT/vio_'+str(i)+'.txt'
				file_obj = open(file_name,'r')
				vif = np.loadtxt(file_obj)
				if vif.shape[0] == 336:# avoiding hd videos
				    continue
				self.X_train.append(vif)
				self.Y_train.append(1)
			except:
				continue
		for i in self.nonviolence_indexes:
			try:
			    file_name = 'violent_features_NON_VIOLENT/nonvio_'+str(i)+'.txt'
			    file_obj = open(file_name,'r')
			    vif = np.loadtxt(file_obj)
			    if vif.shape[0] == 336:# avoiding hd videos
			        continue
			    self.X_train.append(vif)
			    self.Y_train.append(-1)
			except:
				continue

		self.clf = svm.SVC(kernel = 'linear')
		self.clf.fit(self.X_train,self.Y_train)
		print self.clf

strong_shuffle(violence_indexes)
strong_shuffle(nonviolence_indexes)

for i in violence_indexes[0:40]:
	try:
		file_name = 'violent_features_VIOLENT/vio_'+str(i)+'.txt'
		file_obj = open(file_name,'r')
		vif = np.loadtxt(file_obj)
		if vif.shape[0] == 336:# avoiding hd videos
		    continue
		X_test.append(vif)
		Y_test.append(1)
	except:
		continue
for i in nonviolence_indexes[0:40]:
	try:
	    file_name = 'violent_features_NON_VIOLENT/nonvio_'+str(i)+'.txt'
	    file_obj = open(file_name,'r')
	    vif = np.loadtxt(file_obj)
	    if vif.shape[0] == 336:# avoiding hd videos
	        continue
	    X_test.append(vif)
	    Y_test.append(-1)
	except:
		continue

total_accuracy = 0.0
for i in range(0,20):
	clf_obj = Classifier()
	pred = []

	for j in X_test:
		pred.append(clf_obj.clf.predict(j.reshape(1,-1)))

	count = 0

	for j in range(0,len(Y_test)):
	    if pred[j][0] == Y_test[j]:
	        count = count + 1	

	total_accuracy += float(count)/len(Y_test)
	print 'accuracy is : '+str(float(count)/len(Y_test))

print 'average accuracy is : ' + str(total_accuracy/20)
