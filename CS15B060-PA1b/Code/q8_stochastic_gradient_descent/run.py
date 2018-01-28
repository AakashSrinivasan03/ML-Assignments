from sklearn import datasets, linear_model #######q8 implemented using Stochastic Gradient Descent code.......
from sklearn.metrics import mean_squared_error, r2_score
from numpy import genfromtxt
from sklearn import metrics
import numpy as np
import sklearn
training_set = genfromtxt('../../Dataset/DS2-train-q8-normalised.csv', delimiter=',') #loading datasets
testing_set= genfromtxt('../../Dataset/DS2-test-q8-normalised.csv', delimiter=',') #Note:I already shuffled dataset and did featurescaling,normalisation on training and test data and stored it in these corresponding files.
#training_features=np.reshape(training_features,(96,1006))  #########commented lines were used to shuffle the dataset ,feature scale and store as csv files
#training_features=sklearn.preprocessing.scale(training_features)
#training_features=training_features.T
#testing_features=np.reshape(testing_features,(96,80))
#testing_features=testing_features.T
#testing_features=sklearn.preprocessing.scale(testing_features)
#scaler=sklearn.preprocessing.StandardScaler()   ###############Feature scaling and normalisation
#scaler.fit(training_features)
#scaler.fit(testing_features)
#training_features=scaler.transform(training_features)     #########transformed training set
#testing_features= scaler.transform(testing_features)      #########transformed test features


#training_label_1=np.ones(265)
#training_label_2=2*np.ones(239)
#training_label_3=3*np.ones(223)
#training_label_4=4*np.ones(279)
#training_label=np.zeros((1006,4))
#training_label[:265,0]=np.ones(265)
#training_label[265:504,1]=np.ones(239)
#training_label[504:727,2]=np.ones(223)
#training_label[727:1006,3]=np.ones(279)
#training_label=np.append(training_label_1,training_label_2)
#training_label=np.append(training_label,training_label_3)
#training_label=np.append(training_label,training_label_4)
#training_label=np.reshape(training_label,(518,1))
#testing_label=np.zeros((80,4))
#testing_label[:20,0]=np.ones(20)
#testing_label[20:40,1]=np.ones(20)
#testing_label[40:60,2]=np.ones(20)
#testing_label[60:80,3]=np.ones(20)
#training_set=np.append(training_features,training_label,axis=1)


#testing_set=np.append(testing_features,testing_label,axis=1)
#np.random.shuffle(training_set)
#np.savetxt("../../Dataset/DS2-train-q8-normalised.csv",training_set , delimiter=",")  #########saving shuffled,scaled data
#np.random.shuffle(testing_set)
#np.savetxt("../../Dataset/DS2-test-q8-normalised.csv",testing_set , delimiter=",")



training_label=training_set[:,96:]
training_features=training_set[:,:96]
testing_label=testing_set[:,96:]
testing_features=testing_set[:,:96]





def sigmoid (x):    ###########function computing logistic function           
	return 1/(1 + np.exp(-x))
def sigmoid_dash(x):		#############gradient of logistic regression
	return sigmoid(x)*(1-sigmoid(x))
def softmax(x):           ################function computing Softmax
	return np.exp(x) / np.sum(np.exp(x), axis=0)	

############
##########################################training set##################################
hidden_layers=1
inner_nodes=20        ############This was found to be optimal.basically I tested for inner_nodes=5......100 in order of 5's.So 20,25,30 were found to be optimal and hence I used 20
output_nodes=4
i=0;
output=-1*np.ones(1006)

########random initialisation of initial weights
W1=np.random.rand(inner_nodes,96)
W2=np.random.rand(4,inner_nodes)

b1=np.random.rand(inner_nodes,1)
b2=np.random.rand(4,1)
alpha=0.01  #########learning rate
##################
    ############no of training examples-Stochastic gradient descent
num_iter=0	
while num_iter<100 :
	i=0
	while i<1006:	
		theta=np.row_stack((b1,W1.reshape((inner_nodes*96,1)),b2,W2.reshape((4*inner_nodes,1)))) ##########theta includes all W's and B's as Vector	
		


		input=training_features[i,:]    ##########i th input feature
		
		l=np.argmax(training_label[i,:])
	
		input=input.reshape(96,1)
		
		a1=np.dot(W1,input)+b1  #####################Foreward Propogation Part
		
		h1=sigmoid(a1)
		
		a2=np.dot(W2,h1)+b2
		
		h2=softmax(a2)
		

		###################
		
		
		grad_a2=np.zeros((4,1))   #################################Back Propogation part(Just using Formulation I derived in report)
		grad_a2[l,0]=-1
		grad_a2=grad_a2+h2
		
		grad_W2=np.dot(grad_a2,h1.T)
		grad_b2=grad_a2
		grad_h1=np.dot(W2.T,grad_a2)
		grad_a1=grad_h1*sigmoid_dash(a1)
		grad_W1=np.dot(grad_a1,input.T)
		grad_b1=grad_a1
		
		theta_grad=np.row_stack((grad_b1,grad_W1.reshape((inner_nodes*96,1)),grad_b2,grad_W2.reshape((inner_nodes*4,1))))
		theta=theta- alpha*theta_grad
		b1=theta[:inner_nodes,:]
		W1=np.reshape(theta[inner_nodes:inner_nodes+inner_nodes*96,:],(inner_nodes,96))
		b2=theta[inner_nodes+inner_nodes*96:inner_nodes*97+4,:]
		W2=np.reshape(theta[inner_nodes*97+4:,:],(4,inner_nodes))
		
		
		i=i+1
	#output[i]=np.argmax(h2)
	num_iter=num_iter+1
	
print("Parameters:")	
print(theta)

	

#######################################################test set##################################################
i=0
output_test=-1*np.ones(80)
output_org=np.argmax(testing_label,axis=1)

while i<80:                   #####################Performing foreward propogation on test Set using theta obtained

	input=testing_features[i,:]
	
	input=input.reshape(96,1)
	
	a1=np.dot(W1,input)+b1
	
	h1=sigmoid(a1)
	
	a2=np.dot(W2,h1)+b2
	
	h2=softmax(a2)
	output_test[i]=np.argmax(h2)
	
	i=i+1
	target_names = ['class 1', 'class 2', 'class 3','class 4']
#print("==================output_actual\n",)
print metrics.classification_report(output_org, output_test,target_names=target_names)  ###########Precision,recall,F-score calculated

	###########################################################################################################################################
	###########################################################################################################################################
	###########################################################################################################################################
	#######################################                     Part B               #########################################################
print("Part B..............")

hidden_layers=1       ###########part B is similar,except we have have different loss function
inner_nodes=20
output_nodes=4
i=0
output=-1*np.ones(1006)
W1_sum=np.zeros((inner_nodes,96))
W2_sum=np.zeros((4,inner_nodes))
b1_sum=np.zeros((inner_nodes,1))
b2_sum=np.zeros((4,1))
##W1=np.random.rand(inner_nodes,96)
##W2=np.random.rand(4,inner_nodes)
##################
##b1=np.random.rand(inner_nodes,1)
##b2=np.random.rand(4,1)
#theta=[b1,W1.reshape((W1.shape[0],1)),b2,W2.reshape((W2.shape[0],1))]
##theta=np.row_stack((b1,W1.reshape((inner_nodes*96,1)),b2,W2.reshape((4*inner_nodes,1))))
##print(theta.shape)
W1=np.random.rand(inner_nodes,96)
#print(W1)
W1=np.random.rand(inner_nodes,96)
W2=np.random.rand(4,inner_nodes)
##################
b1=np.random.rand(inner_nodes,1)
b2=np.random.rand(4,1)
alpha=0.001
gamma=0.001
num_iter=0
##################
while num_iter<100 :
	i=0
	W1_sum=np.zeros((inner_nodes,96))
	W2_sum=np.zeros((4,inner_nodes))
	b1_sum=np.zeros((inner_nodes,1))
	b2_sum=np.zeros((4,1))
	while i<1006:
	
		#theta=[b1,W1.reshape((W1.shape[0],1)),b2,W2.reshape((W2.shape[0],1))]
		theta=np.row_stack((b1,W1.reshape((inner_nodes*96,1)),b2,W2.reshape((4*inner_nodes,1))))	
		#print(theta.shape)


		input=training_features[i,:]
		
		l=np.argmax(training_label[i,:])
	
		input=input.reshape(96,1)
		#input=np.asmatrix(input)
		#print(input.shape)
		a1=np.dot(W1,input)+b1
		#print(a1.shape)
		#a1=np.reshape(a1,(a1.shape[0],1))
		h1=sigmoid(a1)
		#print(h1.shape)
		a2=np.dot(W2,h1)+b2
		#a2=np.reshape(a2,(a2.shape[0],1))
		h2=softmax(a2)
		#print(h2.shape)
		#print(h2[0])

		###################
		
		#print(l)
		grad_a2=np.zeros((4,1))
		#grad_a2[l,0]=-1
		#print(grad_a2.shape,training_label[i,:].shape)

		current_training_label=np.reshape(training_label[i,:],(4,1))
		#h2_out=np.row_stack((h2,h2,h2,h2))

		#diff_output=h2-current_training_label[i,:]
		#prod=np.zeros((4,1))
		#prod=(-1*h2)*np.sum(h2,axis=0)
		#prod[l,0]=prod[l,0]+h2[l]
		#grad_a2=
		p=0          ###############computing the gradient of loss with respect to a2
		while p<4 :
			q=0
			
			sum=0
			while q<4 :
				flag=0
				if p==q :
					flag=1
				sum=sum+(h2[q,0]-current_training_label[q,0])*((flag*h2[p,0])-(h2[p,0]*h2[q,0]))
				q=q+1
			grad_a2[p]=sum

			p=p+1		
		##grad_a2=-1*(current_training_label-h2)*(softmax(a2)-(softmax(a2)*softmax(a2)))
		#grad_a2=grad_a2+h2
		#print(grad_a2.shape)
		#exit();
		#print(grad_a2.shape,h1.shape)
		grad_W2=np.dot(grad_a2,h1.T)+gamma*2*W2 #+(gamma*2*W2_sum)/(i+1)  ###############################################
		grad_b2=grad_a2  #+(gamma*2*b2_sum)/(i+1)#+ gamma*2*b2
		grad_h1=np.dot(W2.T,grad_a2)
		grad_a1=grad_h1*sigmoid_dash(a1)
		grad_W1=np.dot(grad_a1,input.T)+gamma*2*W1#+(gamma*2*W1_sum)/(i+1) #+gamma*2*W1 ##################################################
		grad_b1=grad_a1#+(gamma*2*b1)/(i+1)#+gamma*2*b1
		#print(grad_W2.shape)
		theta_grad=np.row_stack((grad_b1,grad_W1.reshape((inner_nodes*96,1)),grad_b2,grad_W2.reshape((inner_nodes*4,1))))
		theta=theta- alpha*theta_grad
		b1=theta[:inner_nodes,:]
		W1=np.reshape(theta[inner_nodes:inner_nodes+inner_nodes*96,:],(inner_nodes,96))
		b2=theta[inner_nodes+inner_nodes*96:inner_nodes*97+4,:]
		W2=np.reshape(theta[inner_nodes*97+4:,:],(4,inner_nodes))
		b1_sum=b1_sum+b1
		b2_sum=b2_sum+b2
		W1_sum=W1_sum+W1
		W2_sum=W2_sum+W2
		i=i+1
		
	
	#output[i]=np.argmax(h2)
	
	num_iter=num_iter+1
	####print(output[i-1])
print("Parameters:")	
print(theta)	


	##no of times!!!!
###################

#######################################################test set##################################################
i=0
output_test=-1*np.ones(80)
output_org=np.argmax(testing_label,axis=1)
#print("kkkkkk")
while i<80:

	input=testing_features[i,:]
	
	input=input.reshape(96,1)
	#input=np.asmatrix(input)
	#print(input.shape)
	a1=np.dot(W1,input)+b1
	#print(a1.shape)
	#a1=np.reshape(a1,(a1.shape[0],1))
	h1=sigmoid(a1)
	#print(h1.shape)
	a2=np.dot(W2,h1)+b2
	#a2=np.reshape(a2,(a2.shape[0],1))
	h2=softmax(a2)
	output_test[i]=np.argmax(h2)
	#print("hhhh",i)
	####print(h2)
	####print(output_test[i])
	i=i+1
	target_names = ['class 1', 'class 2', 'class 3','class 4']
#print("==================output_actual\n",)
print metrics.classification_report(output_org, output_test,target_names=target_names)
