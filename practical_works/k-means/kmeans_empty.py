############################### ################################
# K-means exercise for the M2-OSAE Machine Learning lessons
# contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
################################ ################################


import numpy as np


# This Function returns a distance in ndim dimension
# between two points (arguments are tables with all the dimensions) 
def dist_f(dat, cent):
	
	dist = np.sum((dat[:] - cent[:])**2)

	return dist
	

data_type = "2d"

# Usefull data, feel free to add the ones you may need
# for your own implementation of the algorithm

nb_k = 4


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
nb_points_per_center = np.zeros((nb_k))

############################### ################################
#      Main loop, until the new centers do not move anymore
############################### ################################


	
	################################ ################################
	#         Association phase, loop on the data points
	################################ ################################
	
		
	
	################################ ################################
	#           Update phase, calculate the new centers
	################################ ################################
	
	

################################ ################################
#      Save the ending centroid position for visualisation
################################ ################################

np.savetxt("kmeans_output_%s.dat"%(data_type), centers.T, header="%d %d"%(nb_dim,nb_k), comments="")

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
