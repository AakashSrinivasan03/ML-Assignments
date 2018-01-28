from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from numpy import genfromtxt
import numpy as np
import sklearn

dataset = genfromtxt('../../Dataset/DS2_mean.csv', delimiter=',')
i=0
mean_sq=np.zeros(5)
median_sq=np.zeros(5)
f=open("../../Dataset/coeffs-q5.csv",'ab')  #opening files to store datasets
f1=open("../../Dataset/CandC-train1.csv",'ab')
f2=open("../../Dataset/CandC-train2.csv",'ab')
f3=open("../../Dataset/CandC-train3.csv",'ab')
f4=open("../../Dataset/CandC-train4.csv",'ab')
f5=open("../../Dataset/CandC-train5.csv",'ab')
f1_1=open("../../Dataset/CandC-test1.csv",'ab')
f2_1=open("../../Dataset/CandC-test2.csv",'ab')
f3_1=open("../../Dataset/CandC-test3.csv",'ab')
f4_1=open("../../Dataset/CandC-test4.csv",'ab')
f5_1=open("../../Dataset/CandC-test5.csv",'ab')
while i<5 :
	i=i+1



	np.random.shuffle(dataset)
	if i==1 :
		np.savetxt(f1,dataset[:1595,:] , delimiter=",")
		np.savetxt(f1_1,dataset[1595:,:] , delimiter=",")	
	elif i==2 :	
		np.savetxt(f2,dataset[:1595,:] , delimiter=",")
		np.savetxt(f2_1,dataset[1595:,:] , delimiter=",")
	elif i==3 :	
		np.savetxt(f3,dataset[:1595,:] , delimiter=",")
		np.savetxt(f3_1,dataset[1595:,:] , delimiter=",")
	elif i==4 :	
		np.savetxt(f4,dataset[:1595,:] , delimiter=",")
		np.savetxt(f4_1,dataset[1595:,:] , delimiter=",")
	elif i==5 :	
		np.savetxt(f5,dataset[:1595,:] , delimiter=",")
		np.savetxt(f5_1,dataset[1595:,:] , delimiter=",")			

	X_train=dataset[:1595,:122] #1595=0.8*1994
	y_train=dataset[:1595,122]
	X_test=dataset[1595:,:122] #1595=0.8*1994
	y_test=dataset[1595:,122]
	






	regression = linear_model.LinearRegression(fit_intercept=True)
	regression.fit(X_train, y_train)

	# Make predictions using the testing set
	#diabetes_y_pred = regr.predict(diabetes_X_test)

	# The coefficients
	#print("intercept:",regression.intercept_)
	#print('Coefficients: \n', regression.coef_)

	coeffecients=np.append(np.array(regression.intercept_),regression.coef_)
	coeffecients=np.resize(coeffecients,(1,123))
	print('Coefficients: \n', coeffecients)
	if i==1 :
		f=open("../../Dataset/coeffs-q5.csv",'ab')
		np.savetxt(f,coeffecients , delimiter=",")
	else :
		#fd = open('../../Dataset/coeffs-q5.csv','a')
		#fd.write(coeffecients)
		#fd.close()
		np.savetxt(f,coeffecients , delimiter=",")		









	testing_prediction_y = regression.predict(X_test)
	mean_sq[i-1]=mean_squared_error(y_test, testing_prediction_y)
	print("Mean squared error_mean: %.7f"
     % mean_sq[i-1])


#print("Minimum Sq Error:{}").format(np.amin(mean_sq,axis=1))
print("Average Sq Error_mean:{}").format(np.average(mean_sq))	
dataset_median = genfromtxt('../../Dataset/DS2_median.csv', delimiter=',')
i=0
while i<5 :
	i=i+1
	np.random.shuffle(dataset_median)
	X_train=dataset_median[:1595,:122] #1595=0.8*1994
	y_train=dataset_median[:1595,122]
	X_test=dataset_median[1595:,:122] #1595=0.8*1994
	y_test=dataset_median[1595:,122]
	regression1 = linear_model.LinearRegression(fit_intercept=True)
	regression1.fit(X_train, y_train)

	# Make predictions using the testing set
	#diabetes_y_pred = regr.predict(diabetes_X_test)

	# The coefficients
	#print("intercept:",regression1.intercept_)
	#print('Coefficients: \n', regression1.coef_)

	coeffecients=np.append(np.array(regression1.intercept_),regression1.coef_)
	coeffecients=np.resize(coeffecients,(1,123))
	print('Coefficients: \n', coeffecients)
	if i==1 :
		f=open("../../Dataset/coeffs-q5.csv",'ab')
		np.savetxt(f,coeffecients , delimiter=",")
	else :
		#fd = open('../../Dataset/coeffs-q5.csv','a')
		#fd.write(coeffecients)
		#fd.close()
		np.savetxt(f,coeffecients , delimiter=",")



	testing_prediction_y = regression.predict(X_test)
	median_sq[i-1]=mean_squared_error(y_test, testing_prediction_y)
	print("Mean squared error_median: %.7f"
     % median_sq[i-1])
	

#print("Minimum Sq Error:{}").format(np.amin(mean_sq,axis=1))
print("Average Sq Error_median:{}").format(np.average(median_sq))
f.close()
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f1_1.close()
f2_1.close()
f3_1.close()
f4_1.close()
f5_1.close()
