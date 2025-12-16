import numpy as np
import matplotlib.pyplot as plt

nb_p_class = 50

width = np.array([[0.1,0.1,0.05,0.1],[0.2,0.1,0.1,0.07]])
shift = np.array([[0.0,0.4,1.2,0.4],[0.7,0.1,0.4,1.2]])

data = np.zeros((3,nb_p_class*4))

for i in range(0,2):
	for j in range(0,4):
		data[i,j*nb_p_class:(j+1)*nb_p_class:] = np.random.normal(shift[i,j], width[i,j],(nb_p_class,))


ind = np.arange(0,4*nb_p_class)

np.random.shuffle(ind)

f = open("kmeans_input_file_2d.dat", "w")

f.write("2 " + str((4*nb_p_class))+"\n")

for i in range(0,2):
	for j in range(0,4*nb_p_class):
		f.write(str(data[i,j])+" ")
	f.write("\n")

f.close()

plt.plot(data[0,:], data[1,:], '.')

plt.show()



