from sklearn import svm
import numpy as np
import random
import math

data_violent = range(1,130)
data_nonviolent = range(1,130)


X_train = []
Y_train = []

X_test = []
Y_test = []

m = 0
l = 0

for i in data_nonviolent:
    try:
        file_name = 'violent_features_NON_VIOLENT/nonvio_'+str(i)+'.txt'
        file_obj = open(file_name,'r')
        vif = np.loadtxt(file_obj)
        if vif.shape[0] == 336:# avoiding hd videos
            continue
        l += 1
        X_train.append(vif)
        Y_train.append(0)
        file_obj.close()
    except:
        continue
for i in data_violent:
    try:
        file_name = 'violent_features_VIOLENT/vio_'+str(i)+'.txt'
        file_obj = open(file_name,'r')
        vif = np.loadtxt(file_obj)
        if vif.shape[0] == 336:# avoiding hd videos
            continue
        m += 1
        X_train.append(vif)
        Y_train.append(1)
        file_obj.close()
    except:
        continue
for i in data_nonviolent:
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
for i in data_violent:
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

# initial weights done
i = 0
weights = []
for i in range(0,len(Y_train)):
    if Y_train[i] == 1:
        weights.append(1.0/(2*m))
    else:
        weights.append(1.0/(2*l))

def find_nth_smallest(a, n):
    return np.partition(a, n-1)[n-1]

# selected_features and beta_changes from adaBoost algorithm
selected_features = [92, 176, 239, 155, 134, 6, 218, 197, 48, 27, 113, 52, 215, 194, 21, 31, 63, 0, 10, 1]

beta_changes = [0.49764476376266853, 0.5017246335153752, 0.5179410198053325, 0.5285254534815635, 0.5630516080777929, 0.5693116395494333, 0.5737235805010247, 0.5742768179992037, 0.5778426536467475, 0.5887182606051347, 0.7399533747779677, 0.9109363569861147, 0.9125686394142701, 0.9588202212085146, 0.9596774193548288, 0.9747385662089073, 0.9756097560975535, 0.9891490576813158, 0.9909177516671858, 0.9918032786885406]

selected_features = selected_features[0:9]
beta_changes = beta_changes[0:9]


# gettings classifiers back
classifiers = {}
for each_feature in range(0,252):
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

final_classifiers = [classifiers[index] for index in selected_features]

# calculate threshold
threshold = 0.0
for each_beta in beta_changes:
    threshold += math.log(1.0/each_beta)

threshold /= 2

preds = []
for each_set in X_test:
    total = 0.0
    i = 0
    for each_beta in beta_changes:
        total += final_classifiers[i].predict(each_set[selected_features[i]].reshape(1,-1))[0] * math.log(1.0/each_beta)
        i += 1
    if total > threshold:
        preds.append(1)
    else:
        preds.append(0)

count = 0

for i in range(0,len(Y_test)):
    if preds[i] == Y_test[i]:
        count = count + 1

print 'accuracy is : '+str(float(count)/len(Y_test))
