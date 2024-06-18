
# Neural networks pedagogical materials
# The following code is free to use and modify to any extent 
# (with no responsibility of the original author)

# Reference to the author is a courtesy
# Author : David Cornu => david.cornu@utinam.cnrs.fr


from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
import numpy as np
import tensorflow as tf


######################### ##########################
#          Loading data and pre process
######################### ##########################

raw_data = np.loadtxt("../../../data/iris.data")


nb_dat = np.shape(raw_data)[0]
in_dim = np.shape(raw_data)[1] - 1
out_dim = 3

input = raw_data[:,:in_dim]

targ = np.zeros((nb_dat,out_dim))
for i in range(0,nb_dat):
	targ[i,int(raw_data[i,in_dim])] = 1.0


input[:,:] -= np.mean(input[:,:], axis = 0)
input[:,:] /= np.max(np.abs(input[:,:]), axis = 0)

train = input[:100,:]
train_targ = targ[:100,:]

valid = input[100:,:]
valid_targ = targ[100:,:]


######################### ##########################
#         Loading the neural network model
######################### ##########################

model = keras.Sequential()

model.add(layers.Dense(units=8, activation='sigmoid'))
model.add(layers.Dense(units=3, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


model.fit(train, train_targ, batch_size=16, epochs=800, shuffle=True,  validation_split=0.0, validation_data=(valid, valid_targ))

model.evaluate(valid, valid_targ)

pred = model.predict(valid)

matrix = metrics.confusion_matrix(valid_targ.argmax(axis=1), pred.argmax(axis=1))

print (matrix)
















