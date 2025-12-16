import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nb_p_class = 50

width = np.array([[0.1,0.1,0.05,0.1,0.1],[0.2,0.1,0.1,0.07,0.1], [0.04,0.02,0.1,0.08,0.1]])
shift = np.array([[0.0,0.4,1.2,0.4,0.4],[0.7,0.1,0.4,1.2,1.2], [0.1,0.1,0.1,0.1,0.8]])

data = np.zeros((3,nb_p_class*5))

for i in range(0,3):
	for j in range(0,5):
		data[i,j*nb_p_class:(j+1)*nb_p_class:] = np.random.normal(shift[i,j], width[i,j],(nb_p_class,))

ind = np.arange(0,5*nb_p_class)

np.random.shuffle(ind)

f = open("kmeans_input_file_3d.dat", "w")

f.write("3 " + str((5*nb_p_class))+"\n")

for i in range(0,3):
	for j in range(0,5*nb_p_class):
		f.write(str(data[i,j])+" ")
	f.write("\n")

f.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(data[0,:], data[1,:], data[2,:])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()



