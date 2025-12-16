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
nb_points_per_center = np.zeros((nb_k))

############################### ################################
#      Main loop, until the new centers do not move anymore
############################### ################################

# Emulate a do-while loop 
while True:

	l = l + 1
	
	all_dist = 0.0
	
	# Reset the working memory from the previous iteration
	new_centers[:,:] = 0.0
	nb_points_per_center[:] = 0
	
	################################ ################################
	#         Association phase, loop on the data points
	################################ ################################
	
	for i in range(0, nb_dat):
		
		# Find the nearest point
		k_min = 0
		dist_min = dist_f(input_data[:,i], centers[:,0])
		for j in range(1, nb_k):
			dist_temp = dist_f(input_data[:,i], centers[:,j])
			if(dist_temp <= dist_min):
				k_min = j
				dist_min = np.copy(dist_temp)
		
		# Use in advance the new_centers vector for summing the positions
		new_centers[:,k_min] += input_data[:,i]
		
		# Update the number of points associated with this cluster center
		nb_points_per_center[k_min] += 1
		
		all_dist += dist_min
		
	
	################################ ################################
	#           Update phase, calculate the new centers
	################################ ################################
	
	for i in range(0, nb_k):
		if(nb_points_per_center[i] > 0):
			new_centers[:,i] /= nb_points_per_center[i]
	
	# Compute the sum of distances between the centers and the new ones
	cent_mov = 0.0
	for i in range(0, nb_k):
		cent_mov += dist_f(centers[:,i], new_centers[:,i])
	print ("Step :", l, " error :", all_dist/nb_dat, " cent. move :", cent_mov)
	
	# Effectivly move the centers by puting them at the centroids position
	for i in range(0, nb_k):
		if(nb_points_per_center[i] > 0):
			centers[:,i] = new_centers[:,i]
	
	# End the loop if the overall distance is less than a defined epsilon
	# or if too much iteration has been reached
	if(cent_mov < eps or l > 100):
		break

################################ ################################
#      Save the ending centroid position for visualisation
################################ ################################	

np.savetxt("kmeans_output_%s.dat"%(data_type), centers.T, header="%d %d"%(nb_dim,nb_k), comments="")

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
