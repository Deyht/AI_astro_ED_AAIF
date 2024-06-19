

# Neural networks pedagogical materials
# The following code is free to use and modify to any extent 
# (with no responsibility of the original author)

# Reference to the author is a courtesy
# Author : David Cornu => david.cornu@obspm.fr



import numpy as np
import sys, os
sys.path.insert(0,'..')
from ext_module import *


nb_epochs = 3000
control_interv = 50
hid_dim = 8
learn_rate = 0.03
out_dim = 7
beta = 1.0


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

all_input = np.append(raw_data[:,:in_dim], -1.0*np.ones((nb_dat,1)), axis=1)
all_input[:,:-1] -= np.mean(all_input[:,:-1], axis = 0)
all_input[:,:-1] /= np.max(np.abs(all_input[:,:-1]), axis = 0)

test_frac = 0.5

train_by_class = []
test_by_class = []

nb_per_class_train = np.zeros((out_dim), dtype="int")
nb_per_class_test = np.zeros((out_dim), dtype="int")

for i in range(0,out_dim):
	index = np.where(raw_target[:,i].astype("int") == 1)[0]
	test_size = int(test_frac*np.shape(index)[0])
	nb_per_class_test[i] = test_size
	nb_per_class_train[i] = np.shape(index)[0] - test_size
	test_by_class.append(index[:test_size])
	train_by_class.append(index[test_size:])

train_by_class = np.array(train_by_class, dtype="object")
test_by_class = np.array(test_by_class, dtype="object")

relativ_prop = np.clip(nb_per_class_train, 0, 20).astype("float")
relativ_prop /= np.sum(relativ_prop)

nb_train = 100
nb_test = np.sum(nb_per_class_test)

output = np.zeros((nb_train, out_dim))
delta_o = np.zeros((nb_train, out_dim))
hidden = np.zeros((nb_train, hid_dim+1))
delta_h = np.zeros((nb_train, hid_dim+1))
output_test = np.zeros((nb_test, out_dim))
hidden_test = np.zeros((nb_test, hid_dim+1))


input_test = all_input[np.concatenate(test_by_class),:]
targ_test = raw_target[np.concatenate(test_by_class),:]


index = np.arange(0,nb_dat)

######################### ##########################
#          Initialize network weights
######################### ##########################

weights1 = np.zeros((in_dim+1,hid_dim+1))

weights1[:,:-1] = np.random.rand(in_dim+1,hid_dim)*(0.02)-0.01
weights1[in_dim,hid_dim] = 1.0;
weights2 = np.random.rand(hid_dim+1,out_dim)*(0.02)-0.01


######################### ##########################
#                Main training loop
######################### ##########################
######################### ##########################

input = np.zeros((nb_train, in_dim+1))
targ = np.zeros((nb_train, out_dim))

for t in range(0,nb_epochs):

	if ((((t+1)%control_interv) == 0) or (t == 0)):
		print("\n#########################################################################")
		print("\nIteration :", t+1)
		confmat_batch(input_test, hidden_test, output_test, targ_test, weights1, weights2, beta)

	r_class = np.random.choice(np.arange(0,out_dim), size=nb_train, p=relativ_prop).astype("int")
	r_index = (np.random.random(size=nb_train)*nb_per_class_train[r_class]).astype("int")
	for i in range(0, nb_train):
		l_index = train_by_class[r_class[i]][r_index[i]].astype("int")
		input[i] = all_input[l_index,:]
		targ[i] = raw_target[l_index,:]
	
	quad_error = 0.0
	######################### ##########################
	#             Training on all data once
	######################### ##########################

	#Forward phase
	forward_batch(input, hidden, output, weights1, weights2, beta)
	
	#Back-propagation phase
	backprop_batch(input, hidden, delta_h, output, delta_o, targ, weights1, weights2, learn_rate, beta)

	quad_error = 0.5*np.sum((output[:,:] - targ[:,:])**2)
	
	if((t%control_interv) == 0):
		print("\nAverage training dataset quadratic error :", quad_error/nb_dat)

######################### ##########################
######################### ##########################














