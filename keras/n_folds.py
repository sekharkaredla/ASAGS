
total_accuracy = 0.0
iters = 0

data_violent = range(1,130)
data_nonviolent = range(1,130)
random.shuffle(data_violent)
random.shuffle(data_nonviolent)

for i in range(10,131,10):
    X_train = np.empty((0,252))
    Y_train = np.array([])
    X_test = np.empty((0,252))
    Y_test = np.array([])
    iters += 1
    test_set = range(i-10,i)
    for j in test_set:
        try:
            file_name = 'violent_features_NON_VIOLENT/nonvio_'+str(data_nonviolent[j])+'.txt'
            file_obj = open(file_name,'r')
            vif = np.loadtxt(file_obj)
            if vif.shape[0] == 630:# avoiding hd videos
                continue
            X_test = np.vstack(X_test,vif.reshape(-1,vif.shape[0]))
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
            X_test = np.vstack(X_test,vif.reshape(-1,vif.shape[0]))
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
            X_train = np.vstack(X_train,vif.reshape(-1,vif.shape[0]))
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
            X_train = np.vstack(X_train,reshape(-1,vif.shape[0]))
            Y_train = np.append(Y_train,1)
            file_obj.close()
        except:
            continue
            print 'error in reading vio_%d.txt'%j

    
