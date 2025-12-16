############################### ################################
# K-means exercise for the M2-OSAE Machine Learning lessons
# contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
################################ ################################


import numpy as np


# This Function return a distance in ndim dimension
# between two points (arguments are table with all the dimensions) 
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
    # Initialize variables to store the sum of points in each cluster and the number of points per cluster
    sum_points = np.zeros((nb_dim, nb_k))
    nb_points_per_center.fill(0)

    # Association phase: Assign each point to the nearest center
    for i in range(nb_dat):
        distances = [dist_f(input_data[:,i], centers[:,k]) for k in range(nb_k)]
        closest_center = np.argmin(distances)
        sum_points[:, closest_center] += input_data[:, i]
        nb_points_per_center[closest_center] += 1

    # Update phase: Calculate new centers
    for k in range(nb_k):
        if nb_points_per_center[k] > 0:
            new_centers[:, k] = sum_points[:, k] / nb_points_per_center[k]
        else:
            # Handle the case where a cluster has no points
            new_centers[:, k] = centers[:, k]

    # Check for convergence: if the centers do not change, exit the loop
    if np.allclose(centers, new_centers):
        break
    else:
        centers = np.copy(new_centers)

################################ ################################
#      Save the ending centroid position for visualisation
################################ ################################	

np.savetxt("kmeans_output_%s.dat"%(data_type), centers.T, header="%d %d"%(nb_dim,nb_k), comments="")

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
