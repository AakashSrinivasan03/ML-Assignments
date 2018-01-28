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





def sigmoid (x):    ###########function computing logistic function
	return 1/(1 + np.exp(-x))
def sigmoid_dash(x):   #############gradient of logistic regression
	return sigmoid(x)*(1-sigmoid(x))
def softmax(x):    ################function computing Softmax
	return np.exp(x) / np.sum(np.exp(x), axis=0)	
#print("aaa",testing_set.shape,testing_set[0,3])
############
hidden_layers=1
inner_nodes=100   ############This was found to be optimal.basically I tested for inner_nodes=5......100 in order of 5's.So 20,25,30 were found to be optimal and hence I used 20
output_nodes=4
i=0
output=-1*np.ones(1006)
########random initialisation of initial weights
W1=np.random.rand(inner_nodes,96)
W2=np.random.rand(4,inner_nodes)
##################
b1=np.random.rand(inner_nodes,1)
b2=np.random.rand(4,1)
alpha=0.01 
gamma=0.01   
##################
batch_size=10   ################3taking batchsize as 10
num_iter=0
while num_iter<100 :	
	i=0
	while i<1006:
		theta=np.row_stack((b1,W1.reshape((inner_nodes*96,1)),b2,W2.reshape((4*inner_nodes,1)))) ##########theta includes all W's and B's as Vector	
		

		if(i!=1000):   ##########################################Foreward Propogation###############
			input=training_features[i:i+batch_size,:]
			current_training_label=training_label[i:i+batch_size,:]
			
			input=input.T
		else:
			input=training_features[i:,:]
			current_training_label=training_label[i:,:]
			input=input.T
			#input=input.reshape(96,6)	
		#num_iter=0
		l=np.argmax(training_label[i,:])
	
		
		
		a1=np.dot(W1,input)+b1
		
		
		h1=np.zeros((inner_nodes,input.shape[1]))  #########a1,h1,a2,h2 computed using formulations in report
		h2=np.zeros((4,input.shape[1]))
		#a1=np.reshape(a1,(a1.shape[0],1))
		kl=0
		while kl<input.shape[1]:
			h1[:,kl]=sigmoid(a1[:,kl])
			kl=kl+1
		#print(h1.shape)
		a2=np.dot(W2,h1)+b2
		#a2=np.reshape(a2,(a2.shape[0],1))
		kl=0
		#print(h2.shape,a2.shape)
		while kl<input.shape[1]:
			h2[:,kl]=softmax(a2[:,kl])
			kl=kl+1
		#print(h2.shape)
		#print(h2[0])

		###################
		
		#print(l)
		grad_a2=np.zeros((4,input.shape[1]))
		
		###########################################BackPropogation Part
		p=0
		r=0
		grad_a2=np.zeros((4,input.shape[1]))
		#grad_a2[l,0]=-1
		ii=0
		while ii<input.shape[1]:
			l=np.argmax(current_training_label[ii,:])##training lable stored as ...
			grad_a2[l,ii]=-1
			ii=ii+1
		
		grad_a2=grad_a2+h2
			
				

					
		
		grad_W2=np.dot(grad_a2,h1.T) +gamma*2*W2  ############################################### additional term due to regularisation
		grad_b2=grad_a2+ gamma*2*b2
		grad_h1=np.dot(W2.T,grad_a2)
		grad_a1=grad_h1*sigmoid_dash(a1)
		grad_W1=np.dot(grad_a1,input.T) +gamma*2*W1 ##################################################
		grad_b1=grad_a1+gamma*2*b1
		
		grad_W2=grad_W2
		grad_W1=grad_W1
		grad_b2=np.sum(grad_b2,axis=1).reshape(b2.shape[0],1)#/batch_size
		grad_b1=np.sum(grad_b1,axis=1).reshape(b1.shape[0],1)#/batch_size
			

		theta_grad=np.row_stack((grad_b1.reshape((inner_nodes,1)),grad_W1.reshape((inner_nodes*96,1)),grad_b2.reshape((4,1)),grad_W2.reshape((inner_nodes*4,1))))/batch_size ########remove it
		theta=theta- alpha*theta_grad
		b1=theta[:inner_nodes,:]
		W1=np.reshape(theta[inner_nodes:inner_nodes+inner_nodes*96,:],(inner_nodes,96))
		b2=theta[inner_nodes+inner_nodes*96:inner_nodes*97+4,:]
		W2=np.reshape(theta[inner_nodes*97+4:,:],(4,inner_nodes))
		
		
		i=i+batch_size
	#print(h2.shape)
	#output[i]=np.argmax(h2[:,0])
	#print(output[i])
	num_iter=num_iter+1
print("Parameters:")	
print(theta)	


	##no of times!!!!
###################

#######################################################test set##################################################
i=0
output_test=-1*np.ones(80)
output_org=np.argmax(testing_label,axis=1)

while i<80:         ##############Foreward Propogation on testset

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
	#print(h2)
	#print(output_test[i])
	i=i+1
	target_names = ['class 1', 'class 2', 'class 3','class 4']
#print("==================output_actual\n",)
print metrics.classification_report(output_org, output_test,target_names=target_names)  ####Printing result
###########################################################################################################################################
	#######################################                     Part B               #########################################################
print("Part B..............")

hidden_layers=1
inner_nodes=100
output_nodes=4
i=0
output=-1*np.ones(1006)
#W2_org=np.zeros((4,inner_nodes))
#b1_org=np.zeros((inner_nodes,1))
#b2_org=np.zeros((4,1))
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
alpha=0.01##0.001
gamma=0.01
##################
batch_size=10##4
num_iter=0
while num_iter<100 :
	i=0	
	while i<1006:
	#theta=[b1,W1.reshape((W1.shape[0],1)),b2,W2.reshape((W2.shape[0],1))]
		theta=np.row_stack((b1,W1.reshape((inner_nodes*96,1)),b2,W2.reshape((4*inner_nodes,1))))	
		#print(theta.shape)

		if(i!=1000):
			input=training_features[i:i+batch_size,:]
			current_training_label=training_label[i:i+batch_size,:]
			#input=input.reshape(10,96)
			input=input.T
		else:
			input=training_features[i:,:]
			current_training_label=training_label[i:,:]
			input=input.T
			#input=input.reshape(96,6)	
		#num_iter=0
		l=np.argmax(training_label[i,:])
	
		
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
		####grad_a2=np.zeros((4,input.shape[1]))
		#print(input.shape[1])
		#grad_a2[l,0]=-1
		#print(grad_a2.shape,training_label[i,:].shape)

		###current_training_label=np.reshape(training_label[i:,:],(4,1))
		#h2_out=np.row_stack((h2,h2,h2,h2))

		#diff_output=h2-current_training_label[i,:]
		#prod=np.zeros((4,1))
		#prod=(-1*h2)*np.sum(h2,axis=0)
		#prod[l,0]=prod[l,0]+h2[l]
		#grad_a2=
		###########################################iiiiiiiiiiii
		####p=0
		####r=0
		####grad_a2=np.zeros((4,input.shape[1]))
		#grad_a2[l,0]=-1
		####ii=0
		####while ii<input.shape[1]:
			####l=np.argmax(current_training_label[ii,:])##training lable stored as ...
			####grad_a2[l,ii]=-1
			####ii=ii+1
		#l=np.argmax(current_training_label,axis=1)##training lable stored as ....
		#for ii in l:
		#	grad_a2[ii,0]=-1
		####grad_a2=grad_a2+h2
		grad_a2=np.zeros((4,input.shape[1]))
		p=0
		r=0
		#print(input.shape[1])
		while r<input.shape[1]:    ###############computing the gradient of loss with respect to a2
			p=0
			while p<4 :
				q=0
				
				sum_total=0
			
				sum=0
				while q<4 :
					flag=0
					if p==q :
						flag=1
					sum=sum+(h2[q,r]-current_training_label[r,q])*((flag*h2[p,r])-(h2[p,r]*h2[q,r]))
					q=q+1
				grad_a2[p,r]=sum
				p=p+1
			r=r+1		
				

					
		##grad_a2=-1*(current_training_label-h2)*(softmax(a2)-(softmax(a2)*softmax(a2)))
		#grad_a2=grad_a2+h2
		#print(grad_a2.shape)
		#exit();
		#print(grad_a2.shape,h1.shape)
		grad_W2=np.dot(grad_a2,h1.T) +gamma*2*W2  ###############################################
		grad_b2=grad_a2+ gamma*2*b2
		grad_h1=np.dot(W2.T,grad_a2)
		grad_a1=grad_h1*sigmoid_dash(a1)
		grad_W1=np.dot(grad_a1,input.T) +gamma*2*W1 ##################################################
		grad_b1=grad_a1+gamma*2*b1
		#print(grad_W2.shape)
		#print(grad_W1.shape)
		#print(grad_b2.shape)
		#print(grad_b1.shape)
		#print((gamma*2*b1).shape)
		grad_W2=grad_W2#/batch_size
		grad_W1=grad_W1#/batch_size
		grad_b2=np.sum(grad_b2,axis=1).reshape(b2.shape[0],1)#/batch_size
		grad_b1=np.sum(grad_b1,axis=1).reshape(b1.shape[0],1)#/batch_size
		#print(grad_W2.shape)
		#print(grad_W1.shape)
		#print(grad_b2.shape)
		#print(grad_b1.shape)		

		theta_grad=np.row_stack((grad_b1.reshape((inner_nodes,1)),grad_W1.reshape((inner_nodes*96,1)),grad_b2.reshape((4,1)),grad_W2.reshape((inner_nodes*4,1))))/batch_size
		theta=theta- alpha*theta_grad
		b1=theta[:inner_nodes,:]
		W1=np.reshape(theta[inner_nodes:inner_nodes+inner_nodes*96,:],(inner_nodes,96))
		b2=theta[inner_nodes+inner_nodes*96:inner_nodes*97+4,:]
		W2=np.reshape(theta[inner_nodes*97+4:,:],(4,inner_nodes))
		i=i+batch_size
		
	#print(h2.shape)
	#output[i]=np.argmax(h2[:,0])
	#print(output[i])
	
	num_iter=num_iter+1
	
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
	#print(h2)
	#print(output_test[i])
	i=i+1
	target_names = ['class 1', 'class 2', 'class 3','class 4']
#print("==================output_actual\n",)
print metrics.classification_report(output_org, output_test,target_names=target_names)

