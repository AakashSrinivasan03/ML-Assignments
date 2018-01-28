from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from numpy import genfromtxt
from sklearn import metrics
import numpy as np
import sklearn
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import libsvm
from sklearn.svm import SVC
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
#from svmutil import *
#Dataset = genfromtxt('DS3/Iris_dataset.csv', delimiter=',') #loading datasets
training_features = genfromtxt('../../Dataset/DS2/Train_features_4', delimiter=',') #loading datasets
testing_features= genfromtxt('../../Dataset/DS2/Test_features_4', delimiter=',') 
training_features=np.reshape(training_features,(96,1006))  #####reshape of data
training_features=training_features.T
#print("aaaa",training_features[0,87])
testing_features=np.reshape(testing_features,(96,80))
testing_features=testing_features.T
scaler=sklearn.preprocessing.StandardScaler()    ##############Scaling data
scaler.fit(training_features)
#scaler.fit(testing_features)
training_features=scaler.transform(training_features) #############transforming training and test data
testing_features= scaler.transform(testing_features)
training_label=np.zeros(1006)
training_label[:265]=np.zeros(265)
training_label[265:504]=np.ones(239)
training_label[504:727]=2*np.ones(223)
training_label[727:1006]=3*np.ones(279)
testing_label=np.zeros(80)
testing_label[:20]=np.zeros(20)
testing_label[20:40]=np.ones(20)
testing_label[40:60]=2*np.ones(20)
testing_label[60:80]=3*np.ones(20)
class1_training_label=np.zeros(1006)
#class1_training_label[]
########################using one vs one method..
'''print "Linear Kernel"
training_features = training_features.copy(order='C')
testing_features = testing_features.copy(order='C')
C_arr=[0.001,0.003,0.01,0.03,0.1,0.3,1,3]###10,30 to be included later
i=0;
arr_val=np.zeros(8)
splits=StratifiedKFold(n_splits=5,shuffle=True)
while i<8 :
	#j=0;
	sum=0;
	for train_index, test_index in splits.split(training_features, training_label):
		X_train, X_test = training_features[train_index], training_features[test_index]
		y_train, y_test = training_label[train_index], training_label[test_index]
		model=SVC(kernel='linear',C=C_arr[i])
		model.fit(X_train,y_train)
		###svm.libsvm.svm_save_model('svm_model1.model'.encode(), model)
		######joblib.dump(model, 'svm_model1.model')
		pred = model.predict(X_test)
		#print pred[25]
		print metrics.classification_report(y_test, pred)
		sum=sum+sklearn.metrics.recall_score(y_test, pred,labels=[0,1,2,3],average='micro')
	arr_val[i]=sum/5	
	i=i+1	
print "Best value of C:",C_arr[np.argmax(arr_val)],arr_val[np.argmax(arr_val)]	
model=SVC(kernel='linear',C=C_arr[np.argmax(arr_val)])
model.fit(training_features,training_label)
pred = model.predict(testing_features)
print metrics.classification_report(testing_label, pred)
fig = pyplot.figure()
pyplot.plot(C_arr, arr_val)
pyplot.scatter(C_arr, arr_val,c='g')
pyplot.xlabel("Values of C")
pyplot.ylabel("F1-Score")
fig.suptitle('Linear Kernel')
pyplot.show()
print arr_val'''


#libsvm.fit(training_features,training_label)
#############################################Polynomial kernel#################
'''print "Polynomial Kernel"
#C_arr=[0.01,0.03,0.1,0.3,1,3]
degree_arr=[1,2,3,4,5,10]
gamma_arr=[0.01,0.03,0.1,0.3,1,3]
#coef0_arr=[0.01,0.1,1,10]
arr_val=np.zeros((6,6))
i=0;j=0;k=0;l=0;
#c=1,coef0=0.1
splits=StratifiedKFold(n_splits=5,shuffle=True)
j=0
while j<6:
	k=0
	while k<6:
		
		
		sum=0;
		for train_index, test_index in splits.split(training_features, training_label):
			X_train, X_test = training_features[train_index], training_features[test_index]
			y_train, y_test = training_label[train_index], training_label[test_index]
			model=SVC(kernel='poly',degree=degree_arr[j],gamma=gamma_arr[k])
			model.fit(X_train,y_train)
			###svm.libsvm.svm_save_model('svm_model1.model'.encode(), model)
			######joblib.dump(model, 'svm_model1.model')
			pred = model.predict(X_test)
			#print pred[25]
			print metrics.classification_report(y_test, pred)
			sum=sum+sklearn.metrics.recall_score(y_test, pred,labels=[0,1,2,3],average='micro')
		arr_val[j,k]=sum/5
		
		k=k+1
	j=j+1
	
i=0;j=0;k=0;l=0;
max=0	

j=0
while j<6:
	k=0
	while k<6:
		l=0
		
		if(arr_val[j,k]>max):
			max=arr_val[j,k]
			max_idx=[j,k]
			
		k=k+1
	j=j+1

print "Best value of degree,gamma:",degree_arr[max_idx[0]],gamma_arr[max_idx[1]],arr_val[max_idx[0],max_idx[1]]
model=SVC(kernel='poly',degree=degree_arr[max_idx[0]],gamma=gamma_arr[max_idx[1]])
model.fit(training_features,training_label)
pred = model.predict(testing_features)
print metrics.classification_report(testing_label, pred)

fig = pyplot.figure()
ax  = fig.add_subplot(111, projection = '3d')
#ax.scatter(x_axis_train_label1, y_axis_train_label1, z_axis_train_label1,c='r')
x=np.ones(36)
x[:6]=1*x[:6]
x[6:12]=2*x[6:12]
x[12:18]=3*x[12:18]
x[18:24]=4*x[18:24]
x[24:30]=5*x[24:30]
x[30:]=10*x[30:]
#y=[gamma_arr,gamma_arr,gamma_arr,gamma_arr,gamma_arr,gamma_arr]
y=[0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3]
#y=[]
z=arr_val.reshape(36)
z=z[0:]
print z.shape
print x.shape
#ax.plot_surface(x, y, z)
#ax.plot([1,2], [0.01,0.03], [0.6,0.6])
ax.plot(x, y, z)
ax.scatter(x, y, z,c='b')
prev=[0,0,0]
j=0
while j<6:
	k=0
	while k<6:
		l=0
		if j!=0 or k!=0:
			ax.plot([x[j*6+k],prev[0]], [y[j*6+k],prev[1]], [z[j*6+k][0],prev[2]])
			#ax.plot([x[j*6+k],prev[0]], [y[j*6+k],prev[1]], [z[j*6+k],prev[2]])
		prev=[x[j*6+k],y[j*6+k],z[j*6+k]]
			
		k=k+1
	j=j+1

pyplot.xlabel("Values of degree")
pyplot.ylabel("Values of Gamma")
#pyplot.zlabel("F1-Score")
fig.suptitle('Polynomial Kernel')
pyplot.show()
#ax.plot(x, y, z)
fig = pyplot.figure()
ax  = fig.add_subplot(111, projection = '3d')

ax.scatter(x, y, z,c='r')
#ax.plot_surface(x, y, z)
prev=[0,0,0]
j=0
while j<6:
	k=0
	while k<6:
		l=0
		if j!=0 or k!=0:
			ax.plot([x[j*6+k],prev[0]], [y[j*6+k],prev[1]], [z[j*6+k][0],prev[2]])
			#ax.plot([x[j*6+k],prev[0]], [y[j*6+k],prev[1]], [z[j*6+k],prev[2]])
		prev=[x[j*6+k],y[j*6+k],z[j*6+k]]
			
		k=k+1
	j=j+1

pyplot.xlabel("Values of degree")
pyplot.ylabel("Values of Gamma")
#pyplot.zlabel("F1-Score")
fig.suptitle('Polynomial Kernel')
pyplot.show()
print arr_val'''

#######################################################################################Guassian############################
'''print "Guassian Kernel"
C_arr=[0.01,0.03,0.1,0.3,1,3]
#degree_arr=[1,2,3,4,5,10]
gamma_arr=[0.01,0.03,0.1,0.3,1,3]

arr_val=np.zeros((6,6))
splits=StratifiedKFold(n_splits=5,shuffle=True)
i=0
while i<6:
	#j=0
	
	k=0
	while k<6:
		#l=0
		sum=0;
		for train_index, test_index in splits.split(training_features, training_label):
			X_train, X_test = training_features[train_index], training_features[test_index]
			y_train, y_test = training_label[train_index], training_label[test_index]
			model=SVC(kernel='rbf',C=C_arr[i],gamma=gamma_arr[k])
			model.fit(X_train,y_train)
			###svm.libsvm.svm_save_model('svm_model1.model'.encode(), model)
			######joblib.dump(model, 'svm_model1.model')
			pred = model.predict(X_test)
			#print pred[25]
			print metrics.classification_report(y_test, pred)
			sum=sum+sklearn.metrics.recall_score(y_test, pred,labels=[0,1,2,3],average='micro')
			arr_val[i,k]=sum/5
		
		k=k+1
		
	i=i+1

max=0
i=0	
j=0
k=0
l=0
while i<6:
	j=0
	
	k=0
	while k<6:
	
	
		if(arr_val[i,k]>max):
			max=arr_val[i,k]
			max_idx=[i,k]
		
		k=k+1
		
	i=i+1
print "Best value of C:",C_arr[max_idx[0]],gamma_arr[max_idx[1]],arr_val[max_idx[0],max_idx[1]]	
model=SVC(kernel='rbf',C=C_arr[max_idx[0]],gamma=gamma_arr[max_idx[1]])
model.fit(training_features,training_label)
pred = model.predict(testing_features)
print metrics.classification_report(testing_label, pred)
fig = pyplot.figure()
ax  = fig.add_subplot(111, projection = '3d')
x=np.ones(36)
x[:6]=0.01*x[:6]
x[6:12]=0.03*x[6:12]
x[12:18]=0.1*x[12:18]
x[18:24]=0.3*x[18:24]
x[24:30]=1*x[24:30]
x[30:]=3*x[30:]
y=[0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3]
z=arr_val.reshape(36)
ax.plot(x, y, z)
ax.scatter(x, y, z,c='b')
pyplot.xlabel("Values of C")
pyplot.ylabel("Values of Gamma")
#pyplot.zlabel("F1-Score")
fig.suptitle('Guassian Kernel')
pyplot.show()
fig = pyplot.figure()
ax  = fig.add_subplot(111, projection = '3d')

ax.scatter(x, y, z,c='r')
pyplot.xlabel("Values of C")
pyplot.ylabel("Values of Gamma")
#pyplot.zlabel("F1-Score")
fig.suptitle('Guassian Kernel')
print arr_val
pyplot.show()'''


##########################################################sigmoid################################################
'''print "Sigmoid Kernel"
C_arr=[0.01,0.03,0.1,0.3,1,3]
gamma_arr=[0.01,0.1,1,10]
coef0_arr=[0.01,0.1,1,10]
arr_val=np.zeros((6,4,4))
splits=StratifiedKFold(n_splits=5,shuffle=True)
i=0
while i<6:
	
	
	k=0
	while k<4:
		l=0
		while l<4:
			sum=0;
			for train_index, test_index in splits.split(training_features, training_label):
				X_train, X_test = training_features[train_index], training_features[test_index]
				y_train, y_test = training_label[train_index], training_label[test_index]
				model=SVC(kernel='sigmoid',C=C_arr[i],gamma=gamma_arr[k],coef0=coef0_arr[l])
				model.fit(X_train,y_train)
				###svm.libsvm.svm_save_model('svm_model1.model'.encode(), model)
				######joblib.dump(model, 'svm_model1.model')
				pred = model.predict(X_test)
				#print pred[25]
				print metrics.classification_report(y_test, pred)
				sum=sum+sklearn.metrics.recall_score(y_test, pred,labels=[0,1,2,3],average='micro')
			arr_val[i,k,l]=sum/5
			l=l+1
		k=k+1
		
	i=i+1

max=0
i=0	
j=0
k=0
l=0
while i<6:
	
	
	k=0
	while k<4:
		l=0
		while l<4:
			if(arr_val[i,k,l]>max):
				max=arr_val[i,k,l]
				max_idx=[i,k,l]
			l=l+1
		k=k+1
		
	i=i+1
print "Best value of C:",C_arr[max_idx[0]],arr_val[max_idx[0],max_idx[1],max_idx[2]]	
model=SVC(kernel='sigmoid',C=C_arr[max_idx[0]],gamma=gamma_arr[max_idx[1]],coef0=coef0_arr[max_idx[2]])
model.fit(training_features,training_label)
pred = model.predict(testing_features)
print metrics.classification_report(testing_label, pred)'''	
############################################################################################################		
print "Sigmoidal Kernel"
#C_arr=[0.01,0.03,0.1,0.3,1,3]
degree_arr=[0.01,0.03,0.1,0.3,1,3]
gamma_arr=[0.01,0.03,0.1,0.3,1,3]
#coef0_arr=[0.01,0.1,1,10]
arr_val=np.zeros((6,6))
i=0;j=0;k=0;l=0;
#c=1,coef0=0.1
splits=StratifiedKFold(n_splits=5,shuffle=True)
j=0
while j<6:
	k=0
	while k<6:
		
		
		sum=0;
		for train_index, test_index in splits.split(training_features, training_label):
			X_train, X_test = training_features[train_index], training_features[test_index]
			y_train, y_test = training_label[train_index], training_label[test_index]
			model=SVC(kernel='sigmoid',C=degree_arr[j],gamma=gamma_arr[k])
			model.fit(X_train,y_train)
			###svm.libsvm.svm_save_model('svm_model1.model'.encode(), model)
			######joblib.dump(model, 'svm_model1.model')
			pred = model.predict(X_test)
			#print pred[25]
			print metrics.classification_report(y_test, pred)
			sum=sum+sklearn.metrics.recall_score(y_test, pred,labels=[0,1,2,3],average='micro')
		arr_val[j,k]=sum/5
		
		k=k+1
	j=j+1
	
i=0;j=0;k=0;l=0;
max=0	

j=0
while j<6:
	k=0
	while k<6:
		l=0
		
		if(arr_val[j,k]>max):
			max=arr_val[j,k]
			max_idx=[j,k]
			
		k=k+1
	j=j+1

print "Best value of c,gamma:",degree_arr[max_idx[0]],gamma_arr[max_idx[1]],arr_val[max_idx[0],max_idx[1]]
model=SVC(kernel='sigmoid',C=degree_arr[max_idx[0]],gamma=gamma_arr[max_idx[1]])
model.fit(training_features,training_label)
pred = model.predict(testing_features)
print metrics.classification_report(testing_label, pred)

fig = pyplot.figure()
ax  = fig.add_subplot(111, projection = '3d')
#ax.scatter(x_axis_train_label1, y_axis_train_label1, z_axis_train_label1,c='r')
x=np.ones(36)
x[:6]=0.01*x[:6]
x[6:12]=0.03*x[6:12]
x[12:18]=0.1*x[12:18]
x[18:24]=0.3*x[18:24]
x[24:30]=1*x[24:30]
x[30:]=3*x[30:]
#y=[gamma_arr,gamma_arr,gamma_arr,gamma_arr,gamma_arr,gamma_arr]
y=[0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3,0.01,0.03,0.1,0.3,1,3]
#y=[]
z=arr_val.reshape(36)
z=z[0:]
print z.shape
print x.shape
#ax.plot_surface(x, y, z)
#ax.plot([1,2], [0.01,0.03], [0.6,0.6])
ax.plot(x, y, z)
ax.scatter(x, y, z,c='b')
prev=[0,0,0]
j=0
'''while j<6:
	k=0
	while k<6:
		l=0
		if j!=0 or k!=0:
			ax.plot([x[j*6+k],prev[0]], [y[j*6+k],prev[1]], [z[j*6+k][0],prev[2]])
			#ax.plot([x[j*6+k],prev[0]], [y[j*6+k],prev[1]], [z[j*6+k],prev[2]])
		prev=[x[j*6+k],y[j*6+k],z[j*6+k]]
			
		k=k+1
	j=j+1'''

pyplot.xlabel("Values of C")
pyplot.ylabel("Values of Gamma")
#pyplot.zlabel("F1-Score")
fig.suptitle('Sigmoidal Kernel')
pyplot.show()
#ax.plot(x, y, z)
fig = pyplot.figure()
ax  = fig.add_subplot(111, projection = '3d')

ax.scatter(x, y, z,c='r')
#ax.plot_surface(x, y, z)
prev=[0,0,0]
j=0


pyplot.xlabel("Values of C")
pyplot.ylabel("Values of Gamma")
#pyplot.zlabel("F1-Score")
fig.suptitle('Sigmoidal Kernel')
pyplot.show()
print arr_val








