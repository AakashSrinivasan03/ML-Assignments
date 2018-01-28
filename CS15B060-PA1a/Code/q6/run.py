from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from numpy import genfromtxt
import numpy as np
import sklearn

dataset = genfromtxt('../../Dataset/DS2_mean.csv', delimiter=',')

lambda_vector=np.array([0.01,0.03,0.1,0.3,1,3,10,30])   #computing for different lambdas
average_mean_squared=np.zeros(lambda_vector.shape[0])
i=0;
f=open("../../Dataset/coeffs-q6.csv",'ab')
coeff_mat=np.zeros((1,123),dtype=np.float64)
while i<lambda_vector.shape[0]:
	print("lambda={}").format(lambda_vector[i])
	j=0
	tmp_error=np.zeros(5)
	while j<5:

		np.random.shuffle(dataset)
		X_train=dataset[:1595,:122]
		y_train=dataset[:1595,122]
		X_test=dataset[1595:,:122]
		y_test=dataset[1595:,122]
		ridge = linear_model.Ridge(alpha=lambda_vector[i])
		ridge.fit(X_train, y_train)
		#print("intercept:",ridge.intercept_)
		#print('Coefficients: \n', ridge.coef_)
		coeffecients=np.append(np.array(ridge.intercept_),ridge.coef_)
		coeffecients=np.resize(coeffecients,(1,123))
		np.savetxt(f,coeffecients , delimiter=",")
		testing_prediction_y= ridge.predict(X_test)
		tmp_error[j]=mean_squared_error(y_test, testing_prediction_y)
		#print("hhhh{}").format(j)
		j=j+1
	average_mean_squared[i]=np.average(tmp_error)  	
	print("Mean squared error: %.7f"
	%average_mean_squared[i] ) 
	coeff_mat=np.append(coeff_mat,coeffecients,axis=0)	
	i=i+1
min_idx=np.argmin(average_mean_squared)	
print("Optimal lambda:{}").format(lambda_vector[min_idx])
print(coeff_mat.shape)
coeff_req=coeff_mat[min_idx+1]
print(coeff_req.shape)
i=0
while(i<122):
	if(coeff_req[i]>-0.002 and coeff_req[i]<0.002):
		coeff_req[i]=0
		X_test[:,i]=0*X_test[:,i]
	i=i+1
testing_prediction_y= ridge.predict(X_test)
print("mean_sq_error_reduced_dim:{}").format(mean_squared_error(y_test, testing_prediction_y))

