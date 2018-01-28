from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from numpy import genfromtxt
from sklearn import metrics
import numpy as np
import sklearn
training_features = genfromtxt('../../Dataset/DS2-train-q7.csv', delimiter=',') #loading datasets
testing_features= genfromtxt('../../Dataset/DS2-test-q7.csv', delimiter=',')

training_features=np.reshape(training_features,(96,518))  #####reshape of data
training_features=training_features.T
#print("aaaa",training_features[0,87])
testing_features=np.reshape(testing_features,(96,40))
testing_features=testing_features.T
scaler=sklearn.preprocessing.StandardScaler()    ##############Scaling data
scaler.fit(training_features)
#scaler.fit(testing_features)
training_features=scaler.transform(training_features) #############transforming training and test data
testing_features= scaler.transform(testing_features)

training_label_0=np.zeros(279)
training_label_1=np.ones(239)
training_label=np.append(training_label_0,training_label_1)   
##training_label=np.reshape(training_label,(518,1))
testing_label_0=np.zeros(20)
testing_label_1=np.ones(20)
testing_label=np.append(testing_label_0,testing_label_1)

model = linear_model.LogisticRegression()
model = model.fit(training_features, training_label)  ########fitting logistic regression model

# check the accuracy on the training set
print("training_accuracy:",model.score(training_features, training_label))
#print(testing_label.mean())
predicted = model.predict(testing_features)
print("testset_accuracy:",model.score(testing_features, testing_label))
#print metrics.accuracy_score(testing_label, predicted)
print metrics.classification_report(testing_label, predicted)

#print predicted


