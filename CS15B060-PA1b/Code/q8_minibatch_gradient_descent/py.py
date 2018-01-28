import tensorflow as tf
#a=tf.constant([3,4])
#b=tf.constant([3,4])
#c=tf.add(a,b)

	#file_writer=tf.summary.FileWriter('log_simple_graph',sess.graph)
from sklearn import datasets, linear_model#######q8 implemented using Minibatch Gradient Descent code.......
from sklearn.metrics import mean_squared_error, r2_score
from numpy import genfromtxt
from sklearn import metrics
import numpy as np
import sklearn
#import tensorflow as tf
#from sklearn.neural_network import MLPClassifier
training_set = genfromtxt('../../Dataset/DS2-train-q8-normalised.csv', delimiter=',') #loading datasets
testing_set= genfromtxt('../../Dataset/DS2-test-q8-normalised.csv', delimiter=',')#Note:I already shuffled dataset and did featurescaling,normalisation on training and test data and stored it in these corresponding files.



training_label=training_set[:,96:]
training_features=training_set[:,:96]
testing_label=testing_set[:,96:]
testing_features=testing_set[:,:96]
W1=tf.Variable(np.zeros((20,96)))
b1=tf.Variable(np.zeros( (20,) ))
print training_features[1:2,:].T.shape
pre_act=tf.matmul(W1,training_features[1:2,:].T)+b1
act=tf.sigmoid(pre_act[:,0])	
session=tf.Session()
session.run(tf.initialize_all_variables())
print session.run(act)