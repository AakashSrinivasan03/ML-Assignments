from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from numpy import genfromtxt
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
training = genfromtxt('../../Dataset/DS1-train.csv', delimiter=',')
testing= genfromtxt('../../Dataset/DS1-test.csv', delimiter=',')
#training_y=np.concatenate((np.zeros(1400),np.ones(1400)),axis=0)
#testing_y=np.concatenate((np.zeros(600),np.ones(600)),axis=0)
#print(training_y.shape[0])
#print(training.shape[0])
#np.append(training, training_y,axis=1 ) 
#training=np.c_[training, training_y ] 
#testing=np.c_[testing, testing_y ] 
#np.random.shuffle(training)
#np.random.shuffle(testing)
#np.random.shuffle(training)
#np.random.shuffle(testing)
#training_y=training[:,20]
#testing_y=testing[:,20]
#training=training[:,:20]
#testing=testing[:,:20]
#no=1000
#knn = KNeighborsClassifier(n_neighbors=no)
#knn.fit(training[:,:20], training[:,20])  #################assuming x0

#print("k={}").format(no)
#testing_prediction_y = knn.predict(testing[:,:20])
#print(testing_prediction_y)
#testing_y=testing[:,20]
#print(training[0,:20])
#testing_prediction_y = np.greater_equal(testing_prediction_y_temp,0.5)
#true_positives=np.dot(testing_y,testing_prediction_y.transpose())
#print("true_positives %d",true_positives)
#actual_true=600
#predicted_true=np.dot(np.ones(1200),testing_prediction_y.transpose())
#print("predicted_true %d",predicted_true)
#precision=true_positives/actual_true
#recall=true_positives/predicted_true
#F=(2*precision*recall)/(precision+recall)
#print("precision :{}").format(precision)
#print("recall :{}").format(recall)
#print("F-estimate:{} ").format(F)

x_axis=np.linspace(10,500,num=50)
#print(x_axis)

y_axis=np.zeros(50) #F estimate
precision=np.zeros(50)
recall=np.zeros(50)
F=np.zeros(50)
i=1
while i<=50 :
	no=i*10
	knn = KNeighborsClassifier(n_neighbors=no) #computing k-nearest neighbours in loop,for different k
	knn.fit(training[:,:20], training[:,20])  #################assuming x0

	print("k={}").format(no)
	testing_prediction_y = knn.predict(testing[:,:20])
	#print(testing_prediction_y)
	testing_y=testing[:,20]
	#print(training[0,:20])
	#testing_prediction_y = np.greater_equal(testing_prediction_y_temp,0.5)
	true_positives=np.dot(testing_y,testing_prediction_y.transpose())
	#print("true_positives %d",true_positives)
	actual_true=600
	predicted_true=np.dot(np.ones(1200),testing_prediction_y.transpose())
	#print("predicted_true %d",predicted_true)
	precision[i-1]=true_positives/actual_true
	recall[i-1]=true_positives/predicted_true
	F[i-1]=(2*precision[i-1]*recall[i-1])/(precision[i-1]+recall[i-1])
	accuracy=sklearn.metrics.accuracy_score(testing_y, testing_prediction_y)
	print("accuracy :{}").format(accuracy)
	print("precision :{}").format(precision[i-1])
	print("recall :{}").format(recall[i-1])
	print("F-estimate:{} ").format(F[i-1])
	i=i+1
max_idx=np.argmax(F)	
print("Optimal K:{}").format((max_idx+1)*10)	


plt.plot(x_axis,F)    #plotting k nearest neighbours
plt.ylabel('F estimate')
plt.xlabel('K')
plt.show()




#print(testing[900,20])
#training_set=np.concatenate((training,training_y),axis=1)
#test_set=np.concatenate((training,training_y),axis=1)	
#print(test_set.shape[1])
