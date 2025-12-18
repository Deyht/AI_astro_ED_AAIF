


# Neural networks pedagogical materials
# The following code is free to use and modify to any extent 
# (with no responsibility of the original author)

# Reference to the author is a courtesy
# Author : David Cornu => david.cornu@utinam.cnrs.fr



import numpy as np
import sys, os
sys.path.insert(0,'..')
from ext_module import *


nb_iter = 100
control_interv = 1
hid_dim = 8
learn_rate = 0.1
out_dim = 7
beta = 1.0

nb_train = 915 # over 1115


######################### ##########################
#          Loading data and pre process
######################### ##########################

if(not os.path.isdir("stellar_spectra_data")):
	os.system("wget https://share.obspm.fr/s/ANxKkxAZoKmXzRw/download/stellar_spectra_data.tar.gz")
	os.system("tar -xvzf stellar_spectra_data.tar.gz")

raw_data = np.loadtxt("stellar_spectra_data/train.dat")
raw_target = np.loadtxt("stellar_spectra_data/target.dat")


nb_dat = np.shape(raw_data)[0]
in_dim = np.shape(raw_data)[1]

input = np.append(raw_data[:,:in_dim], -1.0*np.ones((nb_dat,1)), axis=1)
output = np.zeros(out_dim)
hidden = np.zeros(hid_dim+1)

targ = raw_target


input[:,:-1] -= np.mean(input[:,:-1], axis = 0)
input[:,:-1] /= np.max(np.abs(input[:,:-1]), axis = 0)

# split training and test dataset
input_test = input[nb_train:,:]
targ_test = targ[nb_train:,:]

input = input[0:nb_train,:]
targ = targ[0:nb_train,:]

nb_dat = nb_train


index = np.arange(0,nb_dat)

######################### ##########################
#          Initialize network weights
######################### ##########################

weights1 = np.random.rand(in_dim+1,hid_dim)*(0.02)-0.01
weights2 = np.random.rand(hid_dim+1,out_dim)*(0.02)-0.01


######################### ##########################
#                Main training loop
######################### ##########################
######################### ##########################

for t in range(0,nb_iter):

	if (((t%control_interv) == 0)):
		print("\n#########################################################################")
		print("\nIteration :", t)
		confmat(input_test, targ_test, weights1, weights2, beta)

	#no need to shuffle in SGD

	quad_error = 0.0
	for i in range(0,nb_dat):
		
		i_d = int(np.random.rand(1)*nb_dat)
		
		
		#Forward phase
		forward(input[i_d,:], in_dim+1, hidden, hid_dim+1, output, out_dim, weights1, weights2, beta)
		
		#Back-propagation phase
		backprop(input[i_d,:], in_dim+1, hidden, hid_dim+1, output, targ[i_d,:], out_dim, weights1, weights2, learn_rate, beta)

		quad_error += 0.5*sum((output[:] - targ[i_d,:])**2)
	
	if((t%control_interv) == 0):
		print("\nAverage training dataset quadratic error :", quad_error/nb_dat)


######################### ##########################
######################### ##########################














