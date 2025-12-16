
################################ ################################
# perceptron exercise for the M2-OSAE Machine Learning lessons
# contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
################################ ################################


import numpy as np


nb_epochs = 1
learn_rate = 0.1


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


index = np.arange(0,nb_dat)

######################### ##########################
#          Initialize network weights
######################### ##########################

weights = np.random.rand(in_dim+1,out_dim)*(0.02)-0.01


######################### ##########################
#                Main training loop
######################### ##########################
######################### ##########################



	######################### ##########################
	#             Training on all data once
	######################### ##########################
	
	

######################### ##########################
######################### ##########################














