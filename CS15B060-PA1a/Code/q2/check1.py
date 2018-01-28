from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from numpy import genfromtxt
import numpy as np
import sklearn
training_features = genfromtxt('Train_features', delimiter=',') #loading datasets
testing_features= genfromtxt('Test_features', delimiter=',')
print("aaa"+train_features.shape)