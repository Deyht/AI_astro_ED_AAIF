

# Neural networks pedagogical materials
# The following code is free to use and modify to any extent 
# (with no responsibility of the original author)

# Reference to the author is a courtesy
# Author : David Cornu => david.cornu@utinam.cnrs.fr



import numpy as np
from numba import jit
#uncoment numba lines to expect high speed up with on-line and sgd algorithms


######################### ########################## ##########################
	
#                             ON-LINE FUNCTIONS
	
######################### ########################## ##########################

@jit(nopython=True, cache=True, fastmath=False)
def forward(input_vect, in_dim, hidden_vect, hid_dim, output_vect, out_dim, weights1, weights2, beta):
	######################### ##########################
	#        One forward step with a binary neuron
	######################### ##########################
	
	
	for i in range(0,hid_dim-1):
		h = np.sum(weights1[:,i]*input_vect[:])
		hidden_vect[i] = 1.0/(1.0 + np.exp(-beta*h))
	hidden_vect[hid_dim-1] = -1.0
		
	for i in range(0,out_dim):
		h = np.sum(weights2[:,i]*hidden_vect[:])
		output_vect[i] = 1.0/(1.0 + np.exp(-beta*h))
	
	
	
@jit(nopython=True, cache=True, fastmath=False)
def backprop(input_vect, in_dim, hidden_vect, hid_dim, output_vect, targ_vect, out_dim, weights1, weights2, learn_rate, beta):
	######################### ##########################
	#       One backward step with a binary neuron
	######################### ##########################
	
	
	delta_o = beta*(output_vect[:] - targ_vect[:])*output_vect[:]*(1.0 - output_vect[:])
	delta_h = np.zeros(hid_dim-1)
	
	for i in range(hid_dim-1):
		h = np.sum(weights2[i,:]*delta_o[:])
		delta_h[i] = beta*hidden_vect[i]*(1.0-hidden_vect[i])*h
	
	
	for i in range(0, in_dim):
		weights1[i,0:hid_dim-1] -= learn_rate*(delta_h[0:hid_dim-1]*input_vect[i])
	
	for i in range(0, hid_dim):
		weights2[i,:] -= learn_rate*(delta_o[:]*hidden_vect[i])
	


def confmat(input, targ, weights1, weights2, beta):
	######################### ##########################
	# Forward on an epoch and display a confusion matrix
	######################### ##########################
	
	in_dim = np.shape(input)[1]
	hid_dim = np.shape(weights2)[0]
	out_dim = np.shape(targ)[1]
	nb_data = np.shape(input)[0]
	confmatrix = np.zeros((out_dim, out_dim))
	recall = np.zeros(out_dim)
	precis = np.zeros(out_dim)
	output = np.zeros(out_dim)
	hidden = np.zeros(hid_dim)
	
	accu = 0.0
	quad_error = 0.0
	for i in range(0, nb_data):
	
		forward(input[i,:], in_dim, hidden, hid_dim, output, out_dim, weights1, weights2, beta)
		
		quad_error += 0.5*np.sum((output[:] - targ[i,:])**2)
		
		max_a = np.argmax(output)
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
	
	print( "Average test set quadratic : ", quad_error/nb_data)


######################### ########################## ##########################
	
#    						   BATCH FUNCTIONS
	
######################### ########################## ##########################
	
	
def forward_batch(input, hidden, output, weights1, weights2, beta):
	######################### ##########################
	#        One forward step with a binary neuron
	######################### ##########################
	
	hidden[:,:] = np.matmul(input, weights1)
	
	hidden[:,:-1] = 1.0/(1.0 + np.exp(-beta*hidden[:,:-1]))
		
	output[:,:] = np.matmul(hidden, weights2)
		
	output[:,:] = 1.0/(1.0 + np.exp(-beta*output[:,:]))
	

def backprop_batch(input, hidden, delta_h, output, delta_o, targ, weights1, weights2, learn_rate, beta):
	######################### ##########################
	#       One backward step with a binary neuron
	######################### ##########################
	
	delta_o[:,:] = beta*(output[:,:]-targ[:,:])*output[:,:]*(1.0-output[:,:])

	delta_h[:,:] = np.matmul(delta_o, np.transpose(weights2))

	delta_h[:,:-1] = beta*hidden[:,:-1]*(1.0-hidden[:,:-1])*delta_h[:,:-1] 
	delta_h[:,-1] = 0.0
	

	weights2[:,:] -= learn_rate*np.matmul(np.transpose(hidden),delta_o)
	weights1[:,:] -= learn_rate*np.matmul(np.transpose(input),delta_h)
	



def confmat_batch(input, hidden, output, targ, weights1, weights2, beta):
	######################### ##########################
	# Forward on an epoch and display a confusion matrix
	######################### ##########################
	
	in_dim = np.shape(input)[1]
	hid_dim = np.shape(weights2)[0]
	out_dim = np.shape(targ)[1]
	nb_data = np.shape(input)[0]
	confmatrix = np.zeros((out_dim, out_dim))
	recall = np.zeros(out_dim)
	precis = np.zeros(out_dim)
	
	accu = 0.0
	
	forward_batch(input, hidden, output, weights1, weights2, beta)
		
	quad_error = 0.5*np.sum((output - targ)**2)
	
	
	for i in range(0, nb_data):	
		max_a = np.argmax(output[i,:])
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
	
	print( "Average test set quadratic : ", quad_error/nb_data)
























