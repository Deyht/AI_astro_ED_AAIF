############################### ################################
# K-means exercise for the M2-OSAE Machine Learning lessons
# contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
################################ ################################


import numpy as np
from scipy.spatial.distance import cdist


data_type = "2d"

# Usefull data, feel free to add the ones you may need
# for your own implementation of the algorithm

eps = 0.001
nb_k = 4
l = 0


# Read data and skip the header
input_data = np.loadtxt("kmeans_input_file_%s.dat"%(data_type), skiprows=1)

print (np.shape(input_data))

# Extract dimension and datalist size from the loaded table
nb_dim, nb_dat = np.shape(input_data)

# The origin of the centers are selected randomly
# To the position of some points in the dataset
init_pos = np.random.randint(low=0, high=nb_dat, size=nb_k)

centers = input_data[:,init_pos]
new_centers = np.zeros((nb_dim, nb_k))


############################### ################################
#      Main loop, until the new centers do not move anymore
############################### ################################

# Emulate a do-while loop 
while True:

	l = l + 1
	
	################################ ################################
	#         Association phase, loop on the data points
	################################ ################################
	
	min_table = cdist(input_data.T, centers.T)
	pos_min = np.argmin(min_table,axis=1)
	
	all_dist = np.mean(np.min(min_table,axis=1))
	
	
	################################ ################################
	#           Update phase, calculate the new centers
	################################ ################################
	
	for i in range(0, nb_k):
		loc_ids = np.where(pos_min == i)[0]
		new_centers[:,i] = np.mean(input_data[:,loc_ids], axis=1)
	
	
	# Compute the sum of distances between the centers and the new ones
	cent_mov = np.sum(np.linalg.norm(new_centers - centers, axis=0))
	print ("Step :", l, " error :", all_dist, " cent. move :", cent_mov)
	
	# Effectivly move the centers by puting them at the centroids position
	for i in range(0, nb_k):
		loc_ids = np.where(pos_min == i)[0]
		if(np.shape(loc_ids) != 0):
			centers[:,i] = new_centers[:,i]
	
	# End the loop if the overall distance is less than a defined epsilon
	# or if too much iteration has been reached
	if(cent_mov < eps or l > 100):
		break

################################ ################################
#      Save the ending centroid position for visualisation
################################ ################################	

np.savetxt("kmeans_output_%s.dat"%(data_type), centers.T, header="%d %d"%(nb_dim,nb_k), comments="")

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
