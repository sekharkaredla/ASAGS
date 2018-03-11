from sklearn import tree
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
        file_name = 'Ovif_features_NON_VIOLENT/nonvio_'+str(i)+'.txt'
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
        file_name = 'Ovif_features_VIOLENT/vio_'+str(i)+'.txt'
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
        file_name = 'Ovif_features_NON_VIOLENT/nonvio_'+str(i)+'.txt'
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
        file_name = 'Ovif_features_VIOLENT/vio_'+str(i)+'.txt'
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
selected_features = [96, 137, 141, 78, 65, 105, 51, 27, 55, 92, 29, 38, 79, 74, 20, 11, 107, 70, 43, 47, 118, 87, 62, 9, 17, 33, 56, 100, 24, 54, 8, 60, 116, 73, 127, 44, 83, 142, 109, 61, 36, 42, 37, 117, 133, 101, 71, 136, 15, 35, 19, 108, 7, 2, 18, 88, 90, 106, 63, 115, 123, 114, 45, 89, 46, 1, 6, 10, 82, 16, 34, 119, 128, 64, 99, 72, 143, 97, 125, 110, 0, 28, 52, 124, 81, 80, 25, 135, 26, 132]

beta_changes = [0.5245297287078183, 0.5386973180076647, 0.5469953775038577, 0.5667609480152096, 0.581880846873463, 0.5840646879006066, 0.5873517786561213, 0.6028736779086025, 0.6075252676873888, 0.6122039341629791, 0.6169099144438861, 0.6370121267706179, 0.6373458363061802, 0.6380136637095961, 0.6383477817440166, 0.64152871449008, 0.6462389833982387, 0.6469140865286109, 0.648604269293918, 0.6562532219816493, 0.6564240049494807, 0.6591613303036494, 0.6601901612236468, 0.66500829187397, 0.6740308461859021, 0.6810380912515707, 0.6826228134492608, 0.6827990781479093, 0.6859781696053759, 0.6897023246029313, 0.6920160101116419, 0.69451476793249, 0.6991749524011064, 0.7058511203143191, 0.7140418267178844, 0.7175237891585696, 0.7188101861758988, 0.723420233880487, 0.7271261154714643, 0.7291711517760969, 0.729543496985358, 0.7385281385281481, 0.739281074058026, 0.7396577864414131, 0.7396577864414176, 0.7407889033376611, 0.7436231412135039, 0.7468464549804309, 0.7493193945333676, 0.7529463116543001, 0.7592815682838758, 0.7612103935971836, 0.7658568758931521, 0.7678001540662551, 0.7748315103303411, 0.77718774200686, 0.7912578055307891, 0.7914575666332041, 0.7966670394810444, 0.7984773846842909, 0.7996863096571662, 0.8000896458987005, 0.8025134649910357, 0.8029180695847287, 0.8053495167453363, 0.810028169014089, 0.8102321388325286, 0.8145261493279125, 0.8153463668211222, 0.8155515370705164, 0.8182229767968312, 0.8246251703771065, 0.8291960828968266, 0.8304466727438473, 0.8312813497492165, 0.8377759981695564, 0.8430472693896207, 0.843470277714027, 0.8553938553938681, 0.8558225508318068, 0.8575393154486515, 0.8581839213418163, 0.8592592592592754, 0.8716066643364719, 0.8729159379736505, 0.8867747239840428, 0.9006152389967041, 0.9015151515151484, 0.9176316103617033, 0.9303052150925438]

selected_features = selected_features[0:12]
beta_changes = beta_changes[0:12]


# gettings classifiers back
classifiers = {}
for each_feature in range(0,144):
    train_data_X = []
    train_data_Y = []
    i = 0
    for each_set in X_train:
        train_data_X.append(each_set[each_feature])
        train_data_Y.append(Y_train[i])
        i += 1
    clf = tree.DecisionTreeClassifier(max_depth = 1)
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
