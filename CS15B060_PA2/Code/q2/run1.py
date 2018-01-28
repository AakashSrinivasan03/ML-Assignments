from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import genfromtxt
from sklearn import metrics
import numpy as np
import sklearn
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
training = genfromtxt('DS3/../Dataset/train.csv', delimiter=',') #loading datasets
testing= genfromtxt('DS3/../Dataset/test.csv', delimiter=',')
y_train=genfromtxt('DS3/../Dataset/train_labels.csv', delimiter=',')
y_test= genfromtxt('DS3/../Dataset/test_labels.csv', delimiter=',')
#print training.shape,y_train.shape

LDA_analysis=LinearDiscriminantAnalysis()
LDA_analysis.fit(training,y_train)
X_train=LDA_analysis.transform(training)
direction=LDA_analysis.coef_
#direction=Princ_comp_analysis.components_;
print y_train.shape
train_dataset=np.append(X_train,y_train.reshape(y_train.shape[0],1),1)
X_test=LDA_analysis.transform(testing);
print "hello"
print X_test.shape
test_dataset=np.append(X_test,y_test.reshape(y_test.shape[0],1),1)
#np.random.shuffle(train_dataset);
#np.random.shuffle(test_dataset);
X_train=train_dataset[:,:1]
y_train=train_dataset[:,1:]
X_test=test_dataset[:,:1]
y_test=test_dataset[:,1:]
regression = linear_model.LinearRegression(fit_intercept=True)  #linear regression fn
regression.fit(X_train, y_train)  #fitting model



# The coefficients
#print("intercept:",regression.intercept_)
#print('Coefficients: \n', regression.coef_)
#coeffecients=np.concatenate((np.array(regression.intercept_),regression.coef_),axis=1)
coeffecients=np.append(np.array(regression.intercept_),regression.coef_)
#coeffecients=np.resize(coeffecients,(1,21))
print('Coefficients: \n', coeffecients)
#np.savetxt("../../Dataset/coeffs-q2.csv",coeffecients , delimiter=",")
testing_prediction_y_temp = regression.predict(X_test)
testing_prediction_y = np.greater_equal(testing_prediction_y_temp,1.5)+np.ones(testing_prediction_y_temp.shape)
#true_positives=np.dot(testing_y,testing_prediction_y.transpose())
#print("true_positives %d",true_positives)
#actual_true=600
#predicted_true=np.dot(np.ones(1200),testing_prediction_y.transpose())
#print("predicted_true %d",predicted_true)
#recall=true_positives/actual_true
#precision=true_positives/predicted_true
accuracy=sklearn.metrics.accuracy_score(y_test, testing_prediction_y)
print("accuracy :{}").format(accuracy)
print metrics.classification_report(y_test, testing_prediction_y)
print testing_prediction_y
x_axis_train_label1=training[:1000,:1]
y_axis_train_label1=training[:1000,1:2]
z_axis_train_label1=training[:1000,2:3]
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_axis_train_label1, y_axis_train_label1, z_axis_train_label1,c='r')
x_axis_train_label2=training[1000:2000,:1]
y_axis_train_label2=training[1000:2000,1:2]
z_axis_train_label2=training[1000:2000,2:3]
ax.scatter(x_axis_train_label2, y_axis_train_label2, z_axis_train_label2,c='b')
#ax.plot(x, y, z, label='parametric curve')
#zp=pylab.polyval(direction,1,training[:1000,:1],training[1000:2000,1:2])
#z = -(direction[0,0]*training[:,:1]+direction[0,1]*training[:,1:2])/direction[0,2];
#ax.plot_surface(training[:,:1],training[:,1:2],z,color='g')
#pyplot.plot('o', x, plane)
#print direction[0,0]
#ax.plot_surface(X_train[:,:1],X_train[:,1:2], z, alpha=0.2)
##############################################################################plot the direction??????
##################################################################################

print coeffecients
pyplot.show()
pyplot.scatter(X_train[:1000,:],y_train[:1000,:],c='r')

pyplot.scatter(X_train[1000:2000,:],y_train[1000:2000,:],c='b')
y_dash=X_train*coeffecients[1]+coeffecients[0]
pyplot.plot(X_train, y_dash, 'g')
pyplot.show()
x_classifier_boundry=(1.5-coeffecients[0])/coeffecients[1]
pyplot.scatter(X_train[:1000,:],np.zeros(1000),c='r',marker='x')

pyplot.scatter(X_train[1000:2000,:],np.zeros(1000),c='b')
y_dash=np.linspace(-2.2,2.2,10)
x_dash=x_classifier_boundry*np.ones(10)
pyplot.plot(x_dash, y_dash, 'g')
pyplot.show()
print direction
#projected_data_X=training*direction

#F=(2*precision*recall)/(precision+recall)   #computing precision,recall,F no
#print("accuracy :{}").format(accuracy)
#print("precision :{}").format(precision)
#print("recall :{}").format(recall)
#print("F-estimate:{} ").format(F)
