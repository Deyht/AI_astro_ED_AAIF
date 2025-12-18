

import numpy as np
import sys
#sys.path.insert(0,glob.glob('../../src/build/lib.*/')[-1])
import CIANNA as cnn


def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")



nb_train = 100 # over 150


######################### ##########################
#          Loading data and pre process
######################### ##########################

raw_data = np.loadtxt("../../data/iris.data")


nb_dat = np.shape(raw_data)[0]
in_dim = np.shape(raw_data)[1] - 1
out_dim = 3

input = np.append(raw_data[:,:in_dim], -1.0*np.ones((nb_dat,1)), axis=1)


targ = np.zeros((nb_dat,out_dim))
for i in range(0,nb_dat):
	targ[i,int(raw_data[i,in_dim])] = 1.0


input[:,:-1] -= np.mean(input[:,:-1], axis = 0)
input[:,:-1] /= np.max(np.abs(input[:,:-1]), axis = 0)


# split training and test dataset
input_test = input[nb_train:,:]
targ_test = targ[nb_train:,:]

input = input[0:nb_train,:]
targ = targ[0:nb_train,:]


cnn.init(in_dim=i_ar([in_dim]), in_nb_ch=1, out_dim=3, \
		bias=-1.0, b_size=8, comp_meth="C_BLAS")

cnn.create_dataset("TRAIN", size=nb_train    , input=f_ar(input)     , target=f_ar(targ))
cnn.create_dataset("VALID", size=150-nb_train, input=f_ar(input_test), target=f_ar(targ_test))
cnn.create_dataset("TEST" , size=150-nb_train, input=f_ar(input_test), target=f_ar(targ_test))


cnn.dense(nb_neurons=8, activation="LOGI")
cnn.dense(nb_neurons=3, activation="SMAX")

cnn.train(nb_iter=1000, learning_rate=0.02, momentum=0.8, confmat=1, control_interv=100, save_every=100, silent=2)

cnn.perf_eval()














