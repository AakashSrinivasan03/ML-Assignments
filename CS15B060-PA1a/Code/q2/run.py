from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from numpy import genfromtxt
import numpy as np
import sklearn
training = genfromtxt('../../Dataset/DS1-train.csv', delimiter=',') #loading datasets
testing= genfromtxt('../../Dataset/DS1-test.csv', delimiter=',')
#coeffecients= genfromtxt('../../Dataset/coeffs-q1.csv', delimiter=',')
#print(training.shape[0])
#training_class0=training[:1400,:]
#training_class1=training[1400:,:]
#print(training_class1.shape[1])
np.random.shuffle(training)
np.random.shuffle(testing)
training_y=training[:,20]
testing_y=testing[:,20]
training=training[:,:20]
testing=testing[:,:20]
#print(training_y.shape[0])  # python by default collumn vectors
# Train the model using the training sets
regression = linear_model.LinearRegression(fit_intercept=True)  #linear regression fn
regression.fit(training, training_y)  #fitting model



# The coefficients
#print("intercept:",regression.intercept_)
#print('Coefficients: \n', regression.coef_)
#coeffecients=np.concatenate((np.array(regression.intercept_),regression.coef_),axis=1)
coeffecients=np.append(np.array(regression.intercept_),regression.coef_)
coeffecients=np.resize(coeffecients,(1,21))
print('Coefficients: \n', coeffecients)
np.savetxt("../../Dataset/coeffs-q2.csv",coeffecients , delimiter=",")
testing_prediction_y_temp = regression.predict(testing)
testing_prediction_y = np.greater_equal(testing_prediction_y_temp,0.5)
true_positives=np.dot(testing_y,testing_prediction_y.transpose())
#print("true_positives %d",true_positives)
actual_true=600
predicted_true=np.dot(np.ones(1200),testing_prediction_y.transpose())
print("predicted_true %d",predicted_true)
recall=true_positives/actual_true
precision=true_positives/predicted_true
accuracy=sklearn.metrics.accuracy_score(testing_y, testing_prediction_y)
F=(2*precision*recall)/(precision+recall)   #computing precision,recall,F no
print("accuracy :{}").format(accuracy)
print("precision :{}").format(precision)
print("recall :{}").format(recall)
print("F-estimate:{} ").format(F)

# The mean squared error
#print("Mean squared error: %.2f"
 #     % mean_squared_error(testing_y, testing_prediction_y))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(testing_y, testing_prediction_y))
