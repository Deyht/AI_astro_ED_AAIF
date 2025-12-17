
################################ ################################
# perceptron exercise for the M2-OSAE Machine Learning lessons
# contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
################################ ################################


import numpy as np

nb_epochs = 5000
learn_rate = 0.005


######################### ##########################
#          Loading data and pre process
######################### ##########################

raw_data = np.loadtxt("../../../data/iris.data")


nb_dat = np.shape(raw_data)[0]
in_dim = np.shape(raw_data)[1] - 1
out_dim = 3

input = np.append(raw_data[:,:in_dim], -1.0*np.ones((nb_dat,1)), axis=1)
output = np.zeros(out_dim)

targ = np.zeros((nb_dat,out_dim))
for i in range(0,nb_dat):
	targ[i,int(raw_data[i,in_dim])] = 1.0



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

	#Test performance of the model every few iterations
	if (((t%200) == 0) or ( t == 1)):
		print("\nIteration :", t)
		accu = 0.0
	
		for n in range(0, nb_dat):
			for i in range(0,out_dim):
				h = 0.0
				for j in range(0,in_dim+1):
					h += weights[j,i]*input[n,j]
				
				if(h > 0):
					output[i] = 1.0
				else:
					output[i] = 0.0
			
			if(np.all(output == targ[n])):
				accu += 1
				
		print ("  Accu %8.2f%%"%(accu/nb_dat*100.0))
		
	
	######################### ##########################
	#             Training on all data once
	######################### ##########################
		
	np.random.shuffle(index)
	input = input[index,:]
	targ = targ[index,:]

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














