import os 
count = 1
for each in open('names.txt','r').readlines():
	filename = each[:-1]
	os.rename(filename,'vio_'+str(count)+'.avi')
	count = count + 1

