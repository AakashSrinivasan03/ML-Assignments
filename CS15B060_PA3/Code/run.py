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

vectorizer = CountVectorizer()
#corpus = ['Subject: This is the first  document.','This is the second second document.','And the third one.','Is this the first document?']
#X = vectorizer.fit_transform(corpus)
#print X.toarray()
data=sklearn.datasets.load_files(container_path="../Dataset/2_NaiveBayes",shuffle=False)
print len(data.data)
X = vectorizer.fit_transform(data.data)
#print X.toarray()[0:500,44]
dataset_X=X.toarray()
###tmpp=np.sum(dataset_X,axis=0)
i=0
###print tmpp.shape
###while i<tmpp.shape[0]:
###	if(tmpp[i]>=1090 and tmpp[i]<=1099):
###		print "wow",i
###	i=i+1	
dataset_X=dataset_X[0:dataset_X.shape[1]-1]
print data.filenames
dataset_y=np.zeros(1099)
i=0;
while i<1099:
	#m=re.search(data.filenames[i], '*legit*')
	#print m
	pattern = re.compile("(.)*legit(.)*")
	tmp= pattern.match(data.filenames[i])
	#print tmp
	if(tmp==None):
		dataset_y[i]=1

	i=i+1
print dataset_y[100:500]
tfidf_transformer = TfidfTransformer()
##dataset_X = tfidf_transformer.fit_transform(dataset_X).toarray()
#print dataset_X[476,500:1000]

##tf_transformer = TfidfTransformer(use_idf=False).fit(dataset_X)
##dataset_X = tf_transformer.transform(dataset_X)
##print dataset_X
#print np.sum(dataset_X,axis=1)
#====================multinomial==========================
splits=StratifiedKFold(n_splits=5,shuffle=False)
for train_index, test_index in splits.split(dataset_X, dataset_y):
	X_train, X_test = dataset_X[train_index], dataset_X[test_index]
	y_train, y_test = dataset_y[train_index], dataset_y[test_index]
	prob_spam=np.sum(y_train)/X_train.shape[0]
	print prob_spam
	prob_not_spam=1- prob_spam
	cumulative_spam_val=np.zeros(X_train.shape[1])
	cumulative_not_spam_val=np.zeros(dataset_X.shape[1])
	i=0
	while i<X_train.shape[0]:
		if(y_train[i]==1):
			cumulative_spam_val=cumulative_spam_val+X_train[i]
			#print "hello"
		else:
			cumulative_not_spam_val=cumulative_not_spam_val+X_train[i]
			#print "hello"
		i=i+1
	total_spam_val=np.sum(cumulative_spam_val)
	total_non_spam_val=np.sum(cumulative_not_spam_val)
	print total_spam_val
	condtln_prob=np.zeros((2,X_train.shape[1]))
	num_spam=np.sum(y_train)
	num_not_spam=X_train.shape[0]-num_spam
	condtln_prob[1]=(cumulative_spam_val+1)/(total_spam_val+X_train.shape[1])
	condtln_prob[0]=(cumulative_not_spam_val+1)/(total_non_spam_val+X_train.shape[1])	
	print np.sum(condtln_prob[1])
	#spam
	score=np.zeros(2)
	y_expected=np.zeros(X_test.shape[0])
	i=0
	while i<X_test.shape[0]:
		score[0]=np.dot(np.log(condtln_prob[0]),X_test[i]) + math.log(prob_not_spam)
		score[1]=np.dot(np.log(condtln_prob[1]),X_test[i]) + math.log(prob_spam)
		if(score[0]<=score[1]):
			y_expected[i]=1
		i=i+1
	print metrics.classification_report(y_test, y_expected)	
i=0
splits=StratifiedKFold(n_splits=5,shuffle=False)
for train_index, test_index in splits.split(dataset_X, dataset_y):
	X_train, X_test = dataset_X[train_index], dataset_X[test_index]
	y_train, y_test = dataset_y[train_index], dataset_y[test_index]
	clf1=sklearn.naive_bayes.MultinomialNB(alpha=1.0)
	clf1.fit(X_train,y_train)
	y_pred=clf1.predict(X_test)
	y_prob_pred=clf1.predict_proba(X_test)
	print y_prob_pred
	print metrics.classification_report(y_test, y_pred)
	print y_test.shape,y_prob_pred[:,0].shape
	presicion,recall,thresholds=sklearn.metrics.precision_recall_curve(y_test, y_prob_pred[:,1], pos_label=None, sample_weight=None)
	if i==0:
		pyplot.plot(presicion, recall, 'g')
		pyplot.title('Precision vs Recall Multinomial case')
		pyplot.xlabel('Recall')
		pyplot.ylabel('Precision')
		#pyplot.label('Precision','Recall')
		pyplot.show()
		i=1




'''prob_spam=np.sum(dataset_y)/1099.0
print prob_spam
prob_not_spam=1- prob_spam
cumulative_spam_val=np.zeros(dataset_X.shape[1])
cumulative_not_spam_val=np.zeros(dataset_X.shape[1])
i=0;
while i<1099:
	if(dataset_y[i]==1):
		cumulative_spam_val=cumulative_spam_val+dataset_X[i]
		#print "hello"
	else:
		cumulative_not_spam_val=cumulative_not_spam_val+dataset_X[i]
		#print "hello"
	i=i+1	
print cumulative_spam_val[10:500]
total_spam_val=np.sum(cumulative_spam_val)
total_non_spam_val=np.sum(cumulative_not_spam_val)
print total_spam_val
condtln_prob=np.zeros((2,dataset_X.shape[1]))
num_spam=np.sum(dataset_y)
condtln_prob[1]=(cumulative_spam_val+1)/(total_spam_val+num_spam)
print np.sum(condtln_prob[1])'''

#print dataset_y[100:500]
####################bernoulli########################3
dataset_X_bernoulli=dataset_X>0
dataset_X_bernoulli=dataset_X_bernoulli.astype(int)
print dataset_X_bernoulli[500:1000,30]
splits=StratifiedKFold(n_splits=5,shuffle=False)
for train_index, test_index in splits.split(dataset_X_bernoulli, dataset_y):
	X_train, X_test = dataset_X_bernoulli[train_index], dataset_X_bernoulli[test_index]
	y_train, y_test = dataset_y[train_index], dataset_y[test_index]
	prob_spam=np.sum(y_train)/X_train.shape[0]
	print prob_spam
	prob_not_spam=1- prob_spam
	cumulative_spam_val=np.zeros(X_train.shape[1])
	cumulative_not_spam_val=np.zeros(dataset_X_bernoulli.shape[1])
	i=0
	while i<X_train.shape[0]:
		if(y_train[i]==1):
			cumulative_spam_val=cumulative_spam_val+X_train[i]
			#print "hello"
		else:
			cumulative_not_spam_val=cumulative_not_spam_val+X_train[i]
			#print "hello"
		i=i+1
	total_spam_val=np.sum(cumulative_spam_val)
	total_non_spam_val=np.sum(cumulative_not_spam_val)
	print total_spam_val
	condtln_prob=np.zeros((2,X_train.shape[1]))
	num_spam=np.sum(y_train)
	num_not_spam=X_train.shape[0]-num_spam
	condtln_prob[1]=(cumulative_spam_val+1)/(total_spam_val+num_spam)
	condtln_prob[0]=(cumulative_not_spam_val+1)/(total_non_spam_val+num_not_spam)	
	print np.sum(condtln_prob[1])
	#spam
	score=np.zeros(2)
	y_expected=np.zeros(X_test.shape[0])
	i=0
	while i<X_test.shape[0]:
		score[0]=np.dot(np.log(condtln_prob[0]),X_test[i])+np.dot(np.log(1-condtln_prob[0]),(1-X_test[i]) ) + math.log(prob_not_spam)
		score[1]=np.dot(np.log(condtln_prob[1]),X_test[i]) +np.dot(np.log(1-condtln_prob[1]),(1-X_test[i]) )+ math.log(prob_spam)
		if(score[0]<=score[1]):
			y_expected[i]=1
		i=i+1
	print metrics.classification_report(y_test, y_expected)	
i=0
splits=StratifiedKFold(n_splits=5,shuffle=False)
for train_index, test_index in splits.split(dataset_X_bernoulli, dataset_y):
	X_train, X_test = dataset_X_bernoulli[train_index], dataset_X_bernoulli[test_index]
	y_train, y_test = dataset_y[train_index], dataset_y[test_index]
	clf1=sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=False, class_prior=None)
	clf1.fit(X_train,y_train)
	y_pred=clf1.predict(X_test)
	y_prob_pred=clf1.predict_proba(X_test)
	print y_prob_pred
	print metrics.classification_report(y_test, y_pred)
	presicion,recall,thresholds=sklearn.metrics.precision_recall_curve(y_test, y_prob_pred[:,1], pos_label=None, sample_weight=None)
	if i==0:
		pyplot.plot(presicion, recall, 'g')
		pyplot.title('Precision vs Recall Binomial case')
		pyplot.xlabel('Recall')
		pyplot.ylabel('Precision')
		#pyplot.label('Precision','Recall')
		pyplot.show()
		i=1


####################################dirichlet#################	



alphas1=np.random.randint(low=1,high=1000,size=dataset_X.shape[1])
alphas0=np.random.randint(low=1,high=1000,size=dataset_X.shape[1])
#alphas1=5*np.ones(dataset_X.shape[1])
#alphas0=alphas1
sum_alphas1=np.sum(alphas1)
sum_alphas0=np.sum(alphas0)
splits=StratifiedKFold(n_splits=5,shuffle=False)
for train_index, test_index in splits.split(dataset_X, dataset_y):
	X_train, X_test = dataset_X[train_index], dataset_X[test_index]
	y_train, y_test = dataset_y[train_index], dataset_y[test_index]
	prob_spam=np.sum(y_train)/X_train.shape[0]
	print prob_spam
	prob_not_spam=1- prob_spam
	cumulative_spam_val=np.zeros(X_train.shape[1])
	cumulative_not_spam_val=np.zeros(dataset_X.shape[1])
	i=0
	while i<X_train.shape[0]:
		if(y_train[i]==1):
			cumulative_spam_val=cumulative_spam_val+X_train[i]
			#print "hello"
		else:
			cumulative_not_spam_val=cumulative_not_spam_val+X_train[i]
			#print "hello"
		i=i+1
		alphas1=(1/(cumulative_not_spam_val+1) )*100	
		alphas0=(1/(cumulative_spam_val+1) )*100
		alphas1=alphas1+((cumulative_spam_val+1) )*100	
		alphas0=alphas0+((cumulative_not_spam_val+1) )*100
		sum_alphas1=np.sum(alphas1)
		sum_alphas0=np.sum(alphas0)
	total_spam_val=np.sum(cumulative_spam_val)
	total_non_spam_val=np.sum(cumulative_not_spam_val)
	print total_spam_val
	condtln_prob=np.zeros((2,X_train.shape[1]))
	num_spam=np.sum(y_train)
	num_not_spam=X_train.shape[0]-num_spam
	condtln_prob[1]=(cumulative_spam_val+alphas1)/(total_spam_val+sum_alphas1)
	condtln_prob[0]=(cumulative_not_spam_val+alphas0)/(total_non_spam_val+sum_alphas0)	
	print np.sum(condtln_prob[1])
	#spam
	score=np.zeros(2)
	y_expected=np.zeros(X_test.shape[0])
	i=0
	prob_1=np.zeros(X_test.shape[0])
	while i<X_test.shape[0]:
		score[0]=np.dot(np.log(condtln_prob[0]),X_test[i]) + math.log(prob_not_spam)
		score[1]=np.dot(np.log(condtln_prob[1]),X_test[i]) + math.log(prob_spam)
		if(score[0]<=score[1]):
			y_expected[i]=1
		i=i+1
		score[1]=score[1]/(score[1]+score[0])
		score[0]=1-score[1]
		prob_1[i-1]=score[0]
	print metrics.classification_report(y_test, y_expected)
	score=np.zeros(2)
	y_expected=np.zeros(X_test.shape[0])

	i=0
	if i==0:
		precision,recall,thresholds=sklearn.metrics.precision_recall_curve(y_test, prob_1, pos_label=1, sample_weight=None)
		pyplot.plot(precision, recall, 'g')
		pyplot.title('Precision vs Recall Dirichelet Prior case')
		pyplot.xlabel('Recall')
		pyplot.ylabel('Precision')
		#pyplot.label('Precision','Recall')
		pyplot.show()
		i=1
	i=0	

	#while i<X_test.shape[0]:
		#score[0]=np.dot(np.log(condtln_prob[0]),X_test[i]) + math.log(prob_not_spam)
		#score[1]=np.dot(np.log(condtln_prob[1]),X_test[i]) + math.log(prob_spam)
	#	tmp1=scipy.stats.dirichlet.logpdf(x=condtln_prob[1],alpha=alphas)
	#	sum_log_gamma=0;
	#	j=0
	#	while j<
	#	sum_log_gamma=np.sum(np.log(math.gamma(alphas)))
        #tmp1=math.log(math.gamma(sum_alphas))-math.log(math.gamma(sum_alphas))
		#print score[1]
	#	i=i+1
	#	if(score[0]<=score[1]):
	#		y_expected[i]=1
	#	i=i+1
	#print metrics.classification_report(y_test, y_expected)	'''

####################################beta##################################
#dataset_X_bernoulli=dataset_X>0
#dataset_X_bernoulli=dataset_X_bernoulli.astype(int)
#print dataset_X_bernoulli[500:1000,30]
i=0
splits=StratifiedKFold(n_splits=5,shuffle=False)
for train_index, test_index in splits.split(dataset_X, dataset_y):
	X_train, X_test = dataset_X[train_index], dataset_X[test_index]
	y_train, y_test = dataset_y[train_index], dataset_y[test_index]
	num_spam=np.sum(y_train)
	num_not_spam=X_train.shape[0]-num_spam
	alpha=2000
	beta=1000
	x=(num_spam+alpha-1)/(num_spam+alpha+beta-2)
	l=[ x,1-x]
	clf1=sklearn.naive_bayes.MultinomialNB(alpha=1.0,  fit_prior=False, class_prior=l)
	clf1.fit(X_train,y_train)
	y_pred=clf1.predict(X_test)
	y_prob_pred=clf1.predict_proba(X_test)
	print y_prob_pred
	print metrics.classification_report(y_test, y_pred)
	presicion,recall,thresholds=sklearn.metrics.precision_recall_curve(y_test, y_prob_pred[:,1], pos_label=None, sample_weight=None)
	if i==0:
		pyplot.plot(presicion, recall, 'g')
		pyplot.title('Precision vs Recall Multinomial prior case')
		pyplot.xlabel('Recall')
		pyplot.ylabel('Precision')
		#pyplot.label('Precision','Recall')
		pyplot.show()
		i=1
		

















