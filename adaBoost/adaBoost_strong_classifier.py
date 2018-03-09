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

final_classifiers = []
beta_changes = []

from sklearn import tree
import numpy as np
import random

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
selected_features = [84, 231, 210, 126, 147, 21, 168, 0, 73, 63, 85, 71, 8, 148, 131, 92, 29, 127, 134, 189, 65, 194, 137, 105, 95, 52, 110, 169, 129, 215, 108, 211, 1, 42, 4, 234, 2, 106, 25, 171, 67, 87, 32, 158, 232, 46, 11, 173, 6, 213, 31, 236, 44, 235, 214, 150, 10, 58, 23, 190, 74, 135, 212, 89, 219, 30, 195, 3, 193, 191, 66, 9, 79, 54, 75, 45, 159, 196, 24, 16, 64, 37, 178, 34, 78, 5, 207, 96, 172, 120, 22, 117, 233, 86, 26, 68, 142, 33, 243, 130]

beta_changes = [0.32877406281661603, 0.33471004002384236, 0.3632686787857678, 0.39239550481943436, 0.4087273054107538, 0.4223422115340964, 0.4343827216985418, 0.4350393700787451, 0.47003376477208547, 0.47238139971817417, 0.48142722117201864, 0.4828989072330817, 0.4833901192504233, 0.48486570981952193, 0.48754330185545053, 0.49150687538659554, 0.49250107127552933, 0.4969914040114578, 0.5017246335153738, 0.5027325023969377, 0.5054749783882403, 0.5064878892733528, 0.5105531996915915, 0.518455725634571, 0.5194861851672292, 0.52103449949051, 0.523103833632958, 0.5259212383780425, 0.5269618588338418, 0.5311385727543555, 0.5337606419414769, 0.535864772170509, 0.5376729127832796, 0.5409989184937525, 0.5426673228346409, 0.5467015345142414, 0.5503734111479268, 0.5512173396674542, 0.5519853450836666, 0.5522927602258142, 0.5525234015155234, 0.5533696729435043, 0.5557595910466971, 0.5563002680965214, 0.5600955556661491, 0.5619612337435804, 0.5646900269541728, 0.5685263947961019, 0.5693116395494333, 0.5745931283905925, 0.5748304446119015, 0.5759388668241988, 0.5795122442809598, 0.5811853720050397, 0.5820631876450941, 0.5831818181818232, 0.5851031553398023, 0.5859853275992873, 0.589282092881763, 0.5935641299374788, 0.5943746503229708, 0.6011339258351163, 0.6046583056053182, 0.6109255357418215, 0.6117538176769973, 0.6123341220039041, 0.6124170567357592, 0.6129978388391556, 0.6153251571678825, 0.6164913366336587, 0.6179096774193491, 0.6225155279503174, 0.62377622377622, 0.6335070349140128, 0.6358939567894735, 0.6373465656829541, 0.6502763885232914, 0.6514935988620147, 0.6545444948801797, 0.6569933396765048, 0.6584836781122649, 0.6588347356723238, 0.6624416631310925, 0.6652677433064259, 0.6658872296327746, 0.6677484571185305, 0.6683697908350568, 0.6711269858193902, 0.6746086863614467, 0.6755037682398791, 0.6786441041019537, 0.6824280807213485, 0.6849602236078219, 0.686864338373777, 0.6903208411970817, 0.6909591110152188, 0.6915065832074201, 0.69223709781904, 0.6944324324324259, 0.6947988754325352]



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
