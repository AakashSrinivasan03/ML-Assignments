import sklearn
import numpy as np
import matplotlib.pyplot as plt
mean0=np.random.rand(20)
mean1=mean0 + 2*np.random.rand(20)
#print(mean0)
#print(mean1)

tmp=np.random.rand(20,20) #for covariance matrix
cov=np.dot(tmp,tmp.transpose())   #computing covariance matrix
#mean=[1,2,1,2,1,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2]
#cov = [[1, 0], [0, 100]]  # diagonal covariance
 
dataset_class0 = np.random.multivariate_normal(mean0, cov, 2000)  #generating datasets


#arr1=np.zeros(2000).T;
#print(arr1.shape[0])

#print(dataset_class0[0,:])  #data stored as collumn vectors....

#print(dataset_class0.shape[0])
dataset_class1 = np.random.multivariate_normal(mean1, cov, 2000)

#print(dataset_class1[0:1,:])

dataset_y_class0=np.resize(np.zeros(2000),(2000,1));
#print(dataset_class0.shape)
#print(dataset_y_class0.shape)
dataset_y_class1=np.resize(np.ones(2000),(2000,1));
dataset_class0=np.concatenate((dataset_class0,dataset_y_class0),axis=1)
#testing_y=np.concatenate((np.zeros(600),np.ones(600)),axis=0)
dataset_class1=np.concatenate((dataset_class1,dataset_y_class1),axis=1)

np.random.shuffle(dataset_class0)
trainingset_class0=dataset_class0[:1400,:] #upper limit is not included in python
testset_class0=dataset_class0[1400:,:]
np.random.shuffle(dataset_class1)    #shuffling datasets
trainingset_class1=dataset_class1[0:1400,:]
testset_class1=dataset_class1[1400:,:]
#print(trainingset_class1.shape[1])
#print(trainingset_class1[0,:])

train=np.concatenate((trainingset_class0,trainingset_class1),axis=0)
#print(train.shape[0])
np.savetxt("../../Dataset/DS1-train.csv",train , delimiter=",")
test=np.concatenate((testset_class0,testset_class1),axis=0)
#print(test.shape[0])
np.savetxt("../../Dataset/DS1-test.csv",test , delimiter=",")
#np.random.shuffle(dataset_class1)
#trainingset_class1=dataset_class1[:1400,:]
#testset_class1=dataset_class1[1400:2000,:]
#print(testset_class1.shape[0])

#dataset1=np.random.multivariate_normal(mean1, cov, 2000).T
#dataset2=np.random.multivariate_normal(mean2, cov, 2000).T
#print(dataset[:,1])
#plt.plot(x, y, 'x')
#plt.axis('equal')
#plt.show()
