//############################### ################################
// K-means exercise for the M2-OSAE Machine Learning lessons
// contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
//############################### ################################


#include <stdio.h>
#include <stdlib.h>
//Type Generic Math
#include <tgmath.h>
#include <time.h>


// This Function return a distance in ndim dimension
// between two points (arguments are table with all the dimensions) 
float dist(float* dat, float* cent, int ndim)
{
	int i;
	float dist;

	dist = 0.0;
	for(i = 0; i < ndim; i++)
		dist = dist + (dat[i]-cent[i])*(dat[i]-cent[i]);
	
	return dist;
}

// Helpful functions for dynamic allocation in regular C

// Classical memory continuous dynamic 1D array
float* c_array(size_t dim_a)
{
	return (float*) calloc(dim_a, sizeof(float));
}

// Memory continuous dynamic 2D array
float** c_array_2d(size_t dim_a, size_t dim_b)
{
	int i;
	float *cont_array, **id_array;
	
	cont_array = (float*) calloc(dim_a * dim_b, sizeof(float));
	id_array = (float**) malloc(dim_a * sizeof(float*));
	
	for(i = 0; i < dim_a; i++)
		id_array[i] = &cont_array[i*dim_b];
	
	return id_array;
}

void free_array_2d(float **array)
{
	if(array[0] != NULL)
		free(array[0]);
	if(array != NULL)
		free(array);
}





int main()
{

	// Usefull data, feel free to add the ones you may need
	// for your own implementation of the algorithm
	float **input_data, **centers, **new_centers;
	int *nb_points_per_center;
	int nb_dim, nb_data;
	int nb_k, rand_p;
	int i, j;
	char *data_type = "2d";
	char file_name[60];
	FILE* f;
	
	srand(time(NULL));

	nb_k = 4;

	// This entry file must be edited to change to other
	// number of dimension. The code must be re-compiled //
	sprintf(file_name,"kmeans_input_file_%s.dat", data_type);
	f = fopen(file_name, "r+");

	
	error = fscanf(f, "%d %d\n", &nb_dim, &nb_data);
	input_data = c_array_2d(nb_data, nb_dim);

	printf("%d %d\n", nb_dim, nb_data);

	// Load all the data
	for(i = 0; i < nb_dim; i++)
		for(j = 0; j < nb_data; j++)
			error = fscanf(f, "%f", &input_data[j][i]);
	
	fclose(f);

	// Allocate the tables according to the dimension
	// gave in the input file
	centers = c_array_2d(nb_k, nb_dim);
	new_centers = c_array_2d(nb_k, nb_dim);
	nb_points_per_center = (int*) calloc(nb_k, sizeof(int));

	// The origin of the centers are selected randomly
	// to the position of some points in the dataset
	for(i = 0; i < nb_k; i++)
	{
		rand_p = (rand()/(float)RAND_MAX)*nb_data;
		for(j = 0; j < nb_dim; j++)
			centers[i][j] = input_data[rand_p][j];
	}

	//############################### ################################
	//     Main loop, until the new centers do not move anymore
	//############################### ################################
	

		//############################### ################################
		//         Association phase, loop on the data points
		//############################### ################################
		
		
		
		

		//############################### ################################
		//           Update phase, calculate the new centers
		//############################### ################################
		
		
		
		
		

	//############################### ################################
	//      Save the ending centroid position for visualization
	//############################### ################################	
	sprintf(file_name,"kmeans_output_%s.dat", data_type);
	f = fopen(file_name, "w+");
	
	fprintf(f, " %d %d \n", nb_dim, nb_k);
	
	for(i = 0; i < nb_k; i++)
	{	
		for(j = 0; j < nb_dim; j++)
			fprintf(f, "%f ", centers[i][j]);
		fprintf(f, "\n");
	}	
	
	fclose(f);
	
	free_array_2d(input_data);
	free_array_2d(centers);
	free_array_2d(new_centers);
	free(nb_points_per_center);

	exit(EXIT_SUCCESS);
	
}








