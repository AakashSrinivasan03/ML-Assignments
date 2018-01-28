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
decision_boundry_flag=False
Dataset = genfromtxt('../../Dataset/iris/Iris_dataset.csv', delimiter=',') #loading datasets
X=Dataset[:,2:4]
print X.shape
y=np.ones(150)         ###############Appending X'i and y'i
y[50:100]=2*np.ones(50)
y[100:150]=3*np.ones(50)
y_train=np.ones(105)
y_test=np.ones(45)
y_train[35:70]=2*np.ones(35)
y_train[70:]=3*np.ones(35)
y_test[15:30]=2*np.ones(15)
y_test[30:]=3*np.ones(15)
X_train_class1=X[:35,:]
X_test_class1=X[35:50,:]
X_train_class2=X[50:85,:]
X_test_class2=X[85:100,:]
X_train_class3=X[100:135,:]
X_test_class3=X[135:,:]
###fig = pyplot.figure()
###ax = fig.add_subplot(111)
###print X.shape
###ax.scatter(X_train_class1[:,0], X_train_class1[:,1],c='r')
###ax.scatter(X_train_class2[:,0], X_train_class2[:,1],c='b')
###ax.scatter(X_train_class3[:,0], X_train_class3[:,1],c='g')
###pyplot.show()
X_train=np.append(X_train_class1,X_train_class2,axis=0)
X_train=np.append(X_train,X_train_class3,axis=0)
X_test=np.append(X_test_class1,X_test_class2,axis=0)
X_test=np.append(X_test,X_test_class3,axis=0)
LDA_analysis=LinearDiscriminantAnalysis(store_covariance=True)  #######Performing LDA,storing covariances
LDA_analysis.fit(X_train,y_train)
X_train_transformed=LDA_analysis.transform(X_train)  ##########transforming X into new space
direction=LDA_analysis.coef_
fig = pyplot.figure()          ##########################visualising in original dataset
ax = fig.add_subplot(111)
print X.shape
ax.scatter(X_train_class1[:,0], X_train_class1[:,1],c='b')
ax.scatter(X_train_class2[:,0], X_train_class2[:,1],c='g')
ax.scatter(X_train_class3[:,0], X_train_class3[:,1],c='r')
x_l1=np.zeros(2)
x_l1[1]= 5
y_l1=LDA_analysis.intercept_[0]+(-direction[0,0]/direction[0,1])*x_l1

##print "intercept"
##print LDA_analysis.intercept_
#pyplot.plot(x_l1, y_l1, 'b')
##pyplot.show()
#direction=Princ_comp_analysis.components_;
#print y_train.shape
#train_dataset=np.append(X_train,y_train.reshape(y_train.shape[0],1),1)
X_test_transformed=LDA_analysis.transform(X_test);      ##########Transforming test data into new space

print X_train_transformed.shape   ########################Visualising the projected datapoints
fig = pyplot.figure()
ax = fig.add_subplot(111)
#print X.shape   
ax.scatter(X_train_transformed[:35,0], X_train_transformed[:35,1],c='b')
ax.scatter(X_train_transformed[35:70,0], X_train_transformed[35:70,1],c='g')
ax.scatter(X_train_transformed[70:105,0], X_train_transformed[70:105,1],c='r')
print X_train.shape,y_train.shape

X_dash1=np.append(X_train,y_train.reshape(y_train.shape[0],1),axis=1)
X_dash2=np.append(X_test,y_test.reshape(y_test.shape[0],1),axis=1)
X_dash=np.append(X_dash1,X_dash2,axis=0)
np.random.shuffle(X_dash)
X_train=X_dash[0:105,:2]
y_train=X_dash[0:105,2]
X_test=X_dash[105:150,:2]
y_test=X_dash[105:150,2]
#np.shuffle()





print LDA_analysis.coef_
pooled_cov=LDA_analysis.covariance_
print pooled_cov
class_means=LDA_analysis.means_

y_pred=LDA_analysis.predict(X_test) ##############decision boundry using meshgrid(sk-learn page)
print metrics.classification_report(y_test, y_pred)
pyplot.show()
if decision_boundry_flag:
	mesh_x_min=np.min(X_test[:,0])-1
	mesh_x_max=np.max(X_test[:,0])+1
	mesh_y_min=np.min(X_test[:,1])-1
	mesh_y_max=np.max(X_test[:,1])+1
	xx, yy = np.meshgrid(np.arange(mesh_x_min, mesh_x_max, 0.03), np.arange(mesh_y_min, mesh_y_max, 0.03))
	Z = LDA_analysis.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	pyplot.figure()
	pyplot.pcolormesh(xx, yy, Z)
	pyplot.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
	pyplot.show()

############################################################

#====================================QDA============================================
QDA_analysis=QuadraticDiscriminantAnalysis(store_covariances=True) ########### QDA,store covariances
QDA_analysis.fit(X_train,y_train)###########fitting
##direction=QDA_analysis.coef_
##print direction
if decision_boundry_flag:
	mesh_x_min=np.min(X_test[:,0])-1    #####Plotting decision boundry
	mesh_x_max=np.max(X_test[:,0])+1
	mesh_y_min=np.min(X_test[:,1])-1
	mesh_y_max=np.max(X_test[:,1])+1
	xx, yy = np.meshgrid(np.arange(mesh_x_min, mesh_x_max, 0.03), np.arange(mesh_y_min, mesh_y_max, 0.03))
	Z = QDA_analysis.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	pyplot.figure()
	pyplot.pcolormesh(xx, yy, Z)
	pyplot.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
	pyplot.show()
y_pred=QDA_analysis.predict(X_test)
print metrics.classification_report(y_test, y_pred)
print QDA_analysis.get_params()
print QDA_analysis.covariances_
class_1_covariance=QDA_analysis.covariances_[0]
class_2_covariance=QDA_analysis.covariances_[1]
class_3_covariance=QDA_analysis.covariances_[2]
print class_3_covariance
#class_1_covariance=QDA_analysis.covariance_[]
#X_train_transformed=QDA_analysis.transform(X_train)
#direction=QDA_analysis.coef
#direction=Princ_comp_analysis.components_;
#print y_train.shape
#train_dataset=np.append(X_train,y_train.reshape(y_train.shape[0],1),1)
#X_test_transformed=QDA_analysis.transform(X_test);

#print X_train_transformed.shape
#fig = pyplot.figure()
#ax = fig.add_subplot(111)
#print X.shape
#ax.scatter(X_train_transformed[:35,0], X_train_transformed[:35,1],c='r')
#ax.scatter(X_train_transformed[35:70,0], X_train_transformed[35:70,1],c='b')
#ax.scatter(X_train_transformed[70:105,0], X_train_transformed[70:105,1],c='g')
#print LDA_analysis.coef_
#pyplot.show()
###########################################RDA##########################################
def RDA_analysis_predict(dataset,class_1_covariance,class_2_covariance,class_3_covariance,pooled_cov,class_means): ########Function to predict discrimant function-classwise,given x
		alpha=0.7                  ##########Setting alpha=0.7
		class_1_covariance=(1-alpha)*pooled_cov+(alpha)*class_1_covariance
		class_2_covariance=(1-alpha)*pooled_cov+(alpha)*class_2_covariance
		class_3_covariance=(1-alpha)*pooled_cov+(alpha)*class_3_covariance
		class_1_cov_inverse=np.linalg.inv(class_1_covariance)
		class_2_cov_inverse=np.linalg.inv(class_2_covariance)
		class_3_cov_inverse=np.linalg.inv(class_3_covariance)
		class_1_cov_det=np.linalg.det(class_1_covariance)
		class_2_cov_det=np.linalg.det(class_2_covariance)
		class_3_cov_det=np.linalg.det(class_3_covariance)
		X_new=dataset
		print class_1_cov_det
		print class_2_cov_det
		print np.log10(class_3_cov_det)
		y_pred=np.zeros(X_new.shape[0])
		#print class_means 
		class_means_1= class_means[0].reshape(class_means[0].shape[0],1)
		class_means_2= class_means[1].reshape(class_means[1].shape[0],1)
		class_means_3= class_means[2].reshape(class_means[2].shape[0],1)
		#print class_means_1.shape
		

		i=0
		while i<X_new.shape[0] :
			X=X_new[i,:].reshape(X_new[i,:].shape[0],1)
			#print class_1_cov_inverse
			#print -0.5*np.dot((X_test_temp.T- class_means_1.T),class_1_cov_inverse),"hh"
			temp1=-0.5*np.dot((X.T- class_means_1.T),class_1_cov_inverse)
			temp2=-0.5*np.dot((X.T- class_means_2.T),class_2_cov_inverse)
			temp3=-0.5*np.dot((X.T- class_means_3.T),class_3_cov_inverse)
			discriminant_1=-0.5*np.log10(class_1_cov_det) +np.dot(temp1,((X)- class_means_1))+np.log10(0.33)
			discriminant_2=-0.5*np.log10(class_2_cov_det) +np.dot(temp2,((X)- class_means_2))+np.log10(0.33)
			discriminant_3=-0.5*np.log10(class_3_cov_det) +np.dot(temp3,((X)- class_means_3))+np.log10(0.33)
			discriminant=[discriminant_1,discriminant_2,discriminant_3]
			y_pred[i]=np.argmax(discriminant)+1

			#print discriminant_1
			i=i+1
		#print metrics.classification_report(y_test, y_pred)
		return y_pred
alpha_arr=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
arr_val=np.zeros(9)
j=0
while j<9 :
	alpha=alpha_arr[j]
	class_1_covariance=(1-alpha)*pooled_cov+(alpha)*class_1_covariance   #######Calculating new covariance,using Regularisation term alpha
	class_2_covariance=(1-alpha)*pooled_cov+(alpha)*class_2_covariance
	class_3_covariance=(1-alpha)*pooled_cov+(alpha)*class_3_covariance
	class_1_cov_inverse=np.linalg.inv(class_1_covariance)
	class_2_cov_inverse=np.linalg.inv(class_2_covariance)
	class_3_cov_inverse=np.linalg.inv(class_3_covariance)
	class_1_cov_det=np.linalg.det(class_1_covariance)
	class_2_cov_det=np.linalg.det(class_2_covariance)
	class_3_cov_det=np.linalg.det(class_3_covariance)
	print class_1_cov_det
	print class_2_cov_det
	print np.log10(class_3_cov_det)
	y_pred=np.zeros(45)
	#print class_means 
	class_means_1= class_means[0].reshape(class_means[0].shape[0],1)
	class_means_2= class_means[1].reshape(class_means[1].shape[0],1)
	class_means_3= class_means[2].reshape(class_means[2].shape[0],1)
	#print class_means_1.shape

	i=0
	while i<45 :
		X_test_temp=X_test[i,:].reshape(X_test[i,:].shape[0],1)
		#print class_1_cov_inverse
		#print -0.5*np.dot((X_test_temp.T- class_means_1.T),class_1_cov_inverse),"hh"
		temp1=-0.5*np.dot((X_test_temp.T- class_means_1.T),class_1_cov_inverse)
		temp2=-0.5*np.dot((X_test_temp.T- class_means_2.T),class_2_cov_inverse)
		temp3=-0.5*np.dot((X_test_temp.T- class_means_3.T),class_3_cov_inverse)
		discriminant_1=-0.5*np.log10(class_1_cov_det) +np.dot(temp1,((X_test_temp)- class_means_1))+np.log10(0.33)
		discriminant_2=-0.5*np.log10(class_2_cov_det) +np.dot(temp2,((X_test_temp)- class_means_2))+np.log10(0.33)
		discriminant_3=-0.5*np.log10(class_3_cov_det) +np.dot(temp3,((X_test_temp)- class_means_3))+np.log10(0.33)
		discriminant=[discriminant_1,discriminant_2,discriminant_3]
		y_pred[i]=np.argmax(discriminant)+1  ########predicting class for test set

		#print discriminant_1
		i=i+1
	print metrics.classification_report(y_test, y_pred)
	arr_val[j]=metrics.f1_score(y_test, y_pred,average='micro')
	j=j+1
#print y_test
fig = pyplot.figure()
pyplot.plot(alpha_arr, arr_val)
pyplot.scatter(alpha_arr, arr_val,c='g')
pyplot.xlabel("Values of alpha")
pyplot.ylabel("F1-Score")
fig.suptitle('Variation of F1-score vs regularisation')
pyplot.show()
#print arr_val
if decision_boundry_flag:
	mesh_x_min=np.min(X_test[:,0])-1
	mesh_x_max=np.max(X_test[:,0])+1
	mesh_y_min=np.min(X_test[:,1])-1
	mesh_y_max=np.max(X_test[:,1])+1
	xx, yy = np.meshgrid(np.arange(mesh_x_min, mesh_x_max, 0.03), np.arange(mesh_y_min, mesh_y_max, 0.03))
	Z = RDA_analysis_predict(np.c_[xx.ravel(), yy.ravel()],class_1_cov_inverse,class_2_covariance,class_3_covariance,pooled_cov,class_means,)
	Z = Z.reshape(xx.shape)
	pyplot.figure()
	pyplot.pcolormesh(xx, yy, Z)
	pyplot.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
	pyplot.show()







