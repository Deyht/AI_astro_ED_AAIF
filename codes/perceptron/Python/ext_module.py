

# Neural networks pedagogical materials
# The following code is free to use and modify to any extent 
# (with no responsibility of the original author)

# Reference to the author is a courtesy
# Author : David Cornu => david.cornu@observatoiredeparis.psl.eu




import numpy as np


def forward(input_vect, output_vect, weights):
	######################### ##########################
	#        One forward step with a binary neuron
	######################### ##########################
	
	in_dim = np.shape(input_vect)[0] 
	#Get the in_dim + 1 dimension from the vector dim
	out_dim = np.shape(output_vect)[0]
		
	for i in range(0,out_dim):
		h = 0.0
		for j in range(0,in_dim): #no +1 as it is included in vector dim
			h += weights[j,i]*input_vect[j]
		
		if(h > 0):
			output_vect[i] = 1.0
		else:
			output_vect[i] = 0.0
			


def backprop(input_vect, output_vect, targ_vect, weights, learn_rate):
	######################### ##########################
	#       One backward step with a binary neuron
	######################### ##########################
	
	in_dim = np.shape(input_vect)[0]
	#Get the in_dim + 1 dimension from the vector dim
	out_dim = np.shape(output_vect)[0]
	
	for i in range(0,in_dim): #no +1 as it is included in vector dim
		for j in range(0,out_dim):
			weights[i,j] -= learn_rate*(output_vect[j]-targ_vect[j])*input_vect[i]



def confmat(input, targ, weights):
	######################### ##########################
	# Forward on an epoch and display a confusion matrix
	######################### ##########################
	
	in_dim = np.shape(input)[1]
	out_dim = np.shape(targ)[1]
	nb_data = np.shape(input)[0]
	confmatrix = np.zeros((out_dim, out_dim))
	recall = np.zeros(out_dim)
	precis = np.zeros(out_dim)
	output = np.zeros(out_dim)
	
	accu = 0.0
	count = 1
	
	for i in range(0, nb_data):
	
		forward(input[i,:], output, weights)
		
		max_a = np.argmax(output[:])
		max_b = np.argmax(targ[i,:])
		confmatrix[max_b, max_a] += 1
		if(max_a == max_b):
			accu += 1
	
	for i in range(0,out_dim):
		recall[i] = 0
		precis[i] = 0
		for j in range(0,out_dim):
			recall[i] += confmatrix[i,j]
			precis[i] += confmatrix[j,i]
		
		if(recall[i] > 0.0):
			recall[i] = confmatrix[i,i] / recall[i] * 100.0
		
		if(precis[i] > 0.0):
			precis[i] = confmatrix[i,i] /precis[i] * 100.0
	
	print ("*****************************************************************")
	print ("Confmat :                                           Recall")
	for i in range(0,out_dim):
		print("         ", end="")
		for j in range(0,out_dim):
			print ("%10d"%confmatrix[i,j], end="")
		print("            %6.2f"%recall[i])
	print ("\n Precision  ", end="")
	for i in range(0,out_dim):
		print ("%10.2f"%precis[i], end="")
	print ("  Accu %8.2f%%"%(accu/nb_data*100.0))
	print ("\n*****************************************************************")
	
	






