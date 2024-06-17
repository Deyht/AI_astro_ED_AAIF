

################################ ################################
# perceptron exercise for the M2-OSAE Machine Learning lessons
# contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
################################ ################################



import numpy as np


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
	
	for n in range(0, nb_data):
			
		for i in range(0,out_dim):
			h = 0.0
			for j in range(0,in_dim): #no +1 as it is included in vect dim
				h += weights[j,i]*input[n,j]
			
			if(h > 0):
				output[i] = 1.0
			else:
				output[i] = 0.0
		
		max_a = np.argmax(output[:])
		max_b = np.argmax(targ[n,:])
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
	
	
	

nb_epochs = 400
learn_rate = 0.05


######################### ##########################
#          Loading data and pre process
######################### ##########################

raw_data = np.loadtxt("../../../data/pima-indians-diabetes.data")


nb_dat = np.shape(raw_data)[0]
in_dim = np.shape(raw_data)[1] - 1
out_dim = 2

input = np.append(raw_data[:,:in_dim], -1.0*np.ones((nb_dat,1)), axis=1)
output = np.zeros(out_dim)

targ = np.zeros((nb_dat,out_dim))
for i in range(0,nb_dat):
	targ[i,int(raw_data[i,in_dim])] = 1.0


i_d = np.where(input[:,0] > 6)
input[i_d,0] = 6
input[:,7] = (input[:,7]-30)%10
i_d = np.where(input[:,7] > 5)
input[i_d,7] = 5



input[:,:-1] -= np.mean(input[:,:-1], axis = 0)
input[:,:-1] /= np.max(np.abs(input[:,:-1]), axis = 0)

index = np.arange(0,nb_dat)

######################### ##########################
#          Initialize network weights
######################### ##########################

weights = np.random.rand(in_dim+1,out_dim)*(0.02)-0.01


######################### ##########################
#                Main training loop
######################### ##########################
######################### ##########################

for t in range(0,nb_epochs):

	if (((t%10) == 0) or ( t == 1)):
		print("\nIteration :", t)
		confmat(input, targ, weights)

	np.random.shuffle(index)
	input = input[index,:]
	targ = targ[index,:]

	######################### ##########################
	#             Training on all data once
	######################### ##########################
	for n in range(0,nb_dat):
		#Forward phase

		for i in range(0,out_dim):
			h = 0.0
			for j in range(0,in_dim+1):
				h += weights[j,i]*input[n,j]
			
			if(h > 0):
				output[i] = 1.0
			else:
				output[i] = 0.0
		
		#Back-propagation phase
		for i in range(0,in_dim+1):
			for j in range(0,out_dim):
				weights[i,j] -= learn_rate*(output[j]-targ[n,j])*input[n,i]


######################### ##########################
######################### ##########################














