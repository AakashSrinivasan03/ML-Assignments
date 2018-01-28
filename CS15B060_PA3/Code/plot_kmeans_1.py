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
x=[1,2,4,5,6,30]
y=[11,10,7.7,8,6,0]

pyplot.plot(x,y)
pyplot.scatter(x,y)

pyplot.title('effect of m keeping eps=0.06 fixed')
pyplot.xlabel('m')
pyplot.ylabel('Purity')
pyplot.show()
#pyplot.plot(presicion, recall, 'g')
		
		#pyplot.label('Precision','Recall')
#pyplot.show()
#pyplot.scatter(clf)
