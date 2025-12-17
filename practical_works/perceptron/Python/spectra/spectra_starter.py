

# Neural networks pedagogical materials
# The following code is free to use and modify to any extent 
# (with no responsibility of the original author)

# Reference to the author is a courtesy
# Author : David Cornu => david.cornu@utinam.cnrs.fr



import numpy as np
import sys, os
sys.path.insert(0,'..')
from ext_module import *


nb_epochs = 2000
control_interv = 10
hid_dim = 8
learn_rate = 0.2
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


######################### ##########################
######################### ##########################














