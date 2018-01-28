from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from numpy import genfromtxt
from sklearn import metrics
import numpy as np
import sklearn
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.datasets
import re
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import sklearn.naive_bayes
import math
import scipy
from scipy.stats import beta
import sklearn.cluster
import matplotlib.cm
from sklearn.cluster import AgglomerativeClustering
X = genfromtxt('Dataset/D31.txt')
print X.shape
dataset_X=X[:,:2]
dataset_y=X[:,2]
##clf=sklearn.cluster.KMeans(n_clusters=100) 
clf = AgglomerativeClustering(n_clusters=32, linkage='ward')
clf.fit(dataset_X)
#purity=(clf.labels_==dataset_y)/3100
print sklearn.metrics.homogeneity_score(dataset_y,clf.labels_)
#y = np.choose(dataset_y, [1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])#.astype(np.float)
pyplot.scatter(dataset_X[:,0],dataset_X[:,1], c=[matplotlib.cm.spectral(float(i) /10) for i in clf.labels_]); 
#pyplot.scatter(dataset_X[:,0],dataset_X[:,1],c=y)
pyplot.show()
#pyplot.plot(presicion, recall, 'g')
		
		#pyplot.label('Precision','Recall')
#pyplot.show()
#pyplot.scatter(clf)
