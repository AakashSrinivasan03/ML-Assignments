from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from numpy import genfromtxt
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
dataset1 = genfromtxt('../../Dataset/problem2_data.csv', delimiter=',')
dataset=dataset1[:,5:128]
#X=dataset[:,5:127]
#y=dataset[:,127]
imp = sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(dataset)
dataset_mean=imp.transform(dataset)
#print(X_mean.shape[0])
#print(X_mean[1,:])
#mean=np.nanmean(X,0)
#print(mean.shape)
#i=0
#j=0
#while i<X_mean.shape[1]:
#	while j<X_mean.shape[0]:
#		if(np.isnan(X_mean[j][i])):
#			X_mean[j][i]=mean[i]
#list_nan=np.argwhere(np.isnan(X))

#while i<list_nan.shape[0] :
#	p,q=list_nan[i]
#	X_mean[p][q]=mean[q]
#print(np.isnan(X_mean))
#numpy.isnan(X_mean).any()
#print(y.shape)
#print(dataset_mean.shape)
#y.reshape((X.shape[0],1))
#y = y[:, np.newaxis]
#print(y.shape)
#np.concatenate((X_mean,y),axis=1)
#print(X_mean.shape)
np.savetxt("../../Dataset/DS2_mean.csv",dataset_mean , delimiter=",")
imp1 = sklearn.preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0)
imp1.fit(dataset)
dataset_median=imp1.transform(dataset)
np.savetxt("../../Dataset/DS2_median.csv",dataset_median , delimiter=",")