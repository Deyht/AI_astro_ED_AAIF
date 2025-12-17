

# Neural networks pedagogical materials
# The following code is free to use and modify to any extent 
# (with no responsibility of the original author)

# Reference to the author is a courtesy
# Author : David Cornu => david.cornu@utinam.cnrs.fr



import numpy as np
import sys
sys.path.insert(0,'..')
from ext_module import *


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

	if (((t%200) == 0)):
		print("\nIteration :", t)
		accu = 0.0
		
		for n in range(0,nb_dat):
			forward(input[n,:], output, weights)

			if(np.all(output == targ[n])):
				accu += 1
				
		print ("  Accu %8.2f%%"%(accu/nb_dat*100.0))

	np.random.shuffle(index)
	input = input[index,:]
	targ = targ[index,:]

	######################### ##########################
	#             Training on all data once
	######################### ##########################
	for i in range(0,nb_dat):
		#Forward phase
		forward(input[i,:], output, weights)
		
		#Back-propagation phase
		backprop(input[i,:], output, targ[i,:], weights, learn_rate)


######################### ##########################
######################### ##########################














