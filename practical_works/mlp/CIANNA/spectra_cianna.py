

import numpy as np
import sys, os
#sys.path.insert(0,glob.glob('../../src/build/lib.*/')[-1])
import CIANNA as cnn


def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")



nb_test = 200
class_balance = [5,40,40,60,60,60,25]
nb_train = np.sum(class_balance) # rebalanced over 1115

######################### ##########################
#          Loading data and pre process
######################### ##########################

if(not os.path.isdir("stellar_spectra_data")):
	os.system("wget https://share.obspm.fr/s/ANxKkxAZoKmXzRw/download/stellar_spectra_data.tar.gz")
	os.system("tar -xzf stellar_spectra_data.tar.gz")

raw_data = np.loadtxt("stellar_spectra_data/train.dat")
raw_target = np.loadtxt("stellar_spectra_data/target.dat")


nb_dat = np.shape(raw_data)[0]
in_dim = np.shape(raw_data)[1]
out_dim = 7

input = np.append(raw_data[:,:in_dim], -1.0*np.ones((nb_dat,1)), axis=1)
targ = raw_target

id_classes = []
for i in range(0, out_dim):
	temp = np.where(np.argmax(targ[:,:], axis=1) == i)
	np.random.shuffle(temp)
	#print (np.shape(temp))
	id_classes.append(temp)


input[:,:-1] -= np.mean(input[:,:-1], axis = 0)
input[:,:-1] /= np.max(np.abs(input[:,:-1]), axis = 0)

# split training and test dataset
input_test = input[915:,:]
targ_test = targ[915:,:]


input_train = np.empty((0,in_dim+1))
targ_train = np.empty((0,out_dim))


for i in range(0,out_dim):
	index_list = id_classes[i][0][:class_balance[i]]
	#print (index_list)
	for j in range(0, len(index_list)):
		input_train = np.append(input_train, np.reshape(input[index_list[j],:],(1,in_dim+1)), axis=0)
		targ_train = np.append(targ_train, np.reshape(targ[index_list[j],:],(1,out_dim)), axis=0)

input = input_train
targ = targ_train


cnn.init(in_dim=i_ar([in_dim]), in_nb_ch=1, out_dim=7, \
		bias=-1.0, b_size=64, comp_meth="C_BLAS")

cnn.create_dataset("TRAIN", size=nb_train, input=f_ar(input)     , target=f_ar(targ))
cnn.create_dataset("VALID", size=nb_test , input=f_ar(input_test), target=f_ar(targ_test))
cnn.create_dataset("TEST" , size=nb_test , input=f_ar(input_test), target=f_ar(targ_test))

#Arch 1
cnn.dense(nb_neurons=8, activation="LOGI")
cnn.dense(nb_neurons=7, activation="SMAX")

#Arch 2
#cnn.dense(nb_neurons=64, activation="RELU")
#cnn.dense(nb_neurons=64, activation="RELU")
#cnn.dense(nb_neurons=7, activation="SMAX")


cnn.train(nb_iter=3000, learning_rate=0.001, momentum=0.8, confmat=1, control_interv=100, save_every=100, silent=2)

cnn.perf_eval()














