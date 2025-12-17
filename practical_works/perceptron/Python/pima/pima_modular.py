

# Neural networks pedagogical materials
# The following code is free to use and modify to any extent 
# (with no responsibility of the original author)

# Reference to the author is a courtesy
# Author : David Cornu => david.cornu@utinam.cnrs.fr



import numpy as np
import sys
sys.path.insert(0,'..')
from ext_module import *


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
	for i in range(0,nb_dat):
		#Forward phase
		forward(input[i,:], output, weights)
		
		#Back-propagation phase
		backprop(input[i,:], output, targ[i,:], weights, learn_rate)


######################### ##########################
######################### ##########################














