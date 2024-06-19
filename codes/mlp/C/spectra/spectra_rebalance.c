


/* 
Neural networks pedagogical materials
The following code is free to use and modify to any extent (with no responsibility of the original author)

Reference to the author is a courtesy
Author : David Cornu => david.cornu@utinam.cnrs.fr
*/




/*
#######################################################################################################################

/!\ /!\                                         /!\ /!\ WARNING /!\ /!\                                         /!\ /!\

Since there is no native matrix multiply operation in C, the following code make use of the OpenBLAS optimized library
To use install OpenBLAS and uncomment the corresponding line in compile.cp

#######################################################################################################################
*/




#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include <time.h>


#include "../ext_module_blas.h"


int main()
{
	int nb_train = 130, nb_test=200, in_dim = 3753, hid_dim = 8, out_dim = 7, nb_epochs = 1500, control_interv=10;
	float **input, **input_test, **targ, **targ_test, **hidden, **hidden_test, **delta_h, **delta_h_test, **output, **output_test, **delta_o, **delta_o_test, **weights1, **weights2, *temp_input, *temp_targ;
	float h, learn_rate = 0.02, rdm, mean, maxval, beta=1.0, quad_error;
	int i, j, k, t, ind, temp_class, max_count[] = {10,20,30,20,20,20,10};
	int *class_count;
	FILE *f = NULL, *f2 = NULL;

	srand(time(NULL));
	
	input = create_2d_table(nb_train,in_dim+1);
	input_test = create_2d_table(nb_test,in_dim+1);
	targ = create_2d_table(nb_train, out_dim);
	targ_test = create_2d_table(nb_test, out_dim);
	hidden = create_2d_table(nb_train, hid_dim+1);
	delta_h = create_2d_table(nb_train, hid_dim+1);
	hidden_test = create_2d_table(nb_test, hid_dim+1);
	delta_h_test = create_2d_table(nb_test, hid_dim+1);
	output =  create_2d_table(nb_train, out_dim);
	delta_o = create_2d_table(nb_train, out_dim);
	output_test =  create_2d_table(nb_test, out_dim);
	delta_o_test = create_2d_table(nb_test, out_dim);
	weights1 = create_2d_table(in_dim+1,hid_dim+1);
	weights2 = create_2d_table(hid_dim+1,out_dim);

	temp_input = (float*) calloc(in_dim+1, sizeof(float));
	temp_targ = (float*) calloc(out_dim, sizeof(float));
	class_count = (int*) calloc(out_dim, sizeof(int));
	
	//######################## ##########################
	//          Loading data and pre process
	//######################## ##########################
	
	//instructions on how to DL the data are provided in the Python versions of this script
	f = fopen("stellar_spectra_data/train.dat", "r+");
	if(f == NULL)
	{
		printf("File not found\n");
		exit(EXIT_FAILURE);
	}
	
	f2 = fopen("stellar_spectra_data/target.dat", "r+");
	if(f2 == NULL)
	{
		printf("File not found\n");
		exit(EXIT_FAILURE);
	}
	
	for(i = 0; i < nb_train*(out_dim); i++)	
		targ[0][i] = 0.0;
	
	for(i = 0; i < nb_test*(out_dim); i++)	
		targ_test[0][i] = 0.0;
	
	//Rebalance dataset
	k = 0;
	for(i = 0; i < 915; i++)
	{
		for(j = 0; j < in_dim; j++)
			fscanf(f, "%f ", &temp_input[j]);
		temp_input[in_dim] = -1.0;
		for(j = 0; j < out_dim; j++)
			fscanf(f2, "%f ",  &temp_targ[j]);
		temp_class = argmax(temp_targ, out_dim);
		if(class_count[temp_class] < max_count[temp_class] && k < nb_train)
		{
			class_count[temp_class] += 1;
			for(j = 0; j < in_dim+1; j++)
				input[k][j] = temp_input[j];
			for(j = 0; j < out_dim; j++)
				targ[k][j] = temp_targ[j];
			k++;
		}
		
	}
	
	
	for(i = 0; i < nb_test; i++)
	{
		for(j = 0; j < in_dim; j++)
			fscanf(f, "%f ", &input_test[i][j]);
		input_test[i][in_dim] = -1.0;
		for(j = 0; j < out_dim; j++)
			fscanf(f2, "%f ",  &targ_test[i][j]);
	}
	
	fclose(f);
	fclose(f2);

	
	for(i = 0; i < in_dim; i++)
	{
		mean = 0.0;
		for(j = 0; j < nb_train; j++)
			mean += input[j][i];
		for(j = 0; j < nb_test; j++)
			mean += input_test[j][i];
		
		mean /= (nb_train+nb_test);
		for(j = 0; j <  nb_train; j++)
			input[j][i] -= mean;
		for(j = 0; j <  nb_test; j++)
			input_test[j][i] -= mean;
		maxval = fabsf(input[0][i]);

		for(j = 0; j < nb_train; j++)
		{
			if(abs(input[j][i] > maxval))
				maxval = fabsf(input[j][i]);
		}
		for(j = 0; j < nb_test; j++)
		{
			if(abs(input_test[j][i] > maxval))
				maxval = fabsf(input_test[j][i]);
		}

		for(j = 0; j < nb_train; j++)
			input[j][i] /= maxval;
		for(j = 0; j < nb_test; j++)
			input_test[j][i] /= maxval;
	}

	

	
	//######################## ##########################
	//          Initialize network weights
	//######################## ##########################

	for(i = 0; i < in_dim+1; i++)
	{
		for(j = 0; j < hid_dim; j++)
			weights1[i][j] = (rand()/(double)RAND_MAX)*(0.02)-0.01;
		weights1[i][hid_dim] = 0.0;
	}	
	weights1[in_dim][hid_dim] = 1.0;
	
	for(i = 0; i < hid_dim+1; i++)
		for(j = 0; j < out_dim; j++)
			weights2[i][j] = (rand()/(double)RAND_MAX)*(0.02)-0.01;

	
	//######################## ##########################
	//                Main training loop
	//######################## ##########################
	//######################## ##########################
	
	for(t = 0; t < nb_epochs; t++)
	{
		if ((t+1)%control_interv == 0 || t == 0)
		{
			printf("\nIteration : %d\n", t+1);
		
			confmat_batch(input_test, in_dim, hidden_test, hid_dim, output_test, targ_test, out_dim, nb_test, weights1, weights2, beta);
		}

		shuffle(input, in_dim+1, targ, out_dim, nb_train);

		quad_error = 0.0;
		//######################## ##########################
		//             Training on all data once
		//######################## ##########################
		//Forward phase
		forward_batch(input, in_dim, hidden, hid_dim, output, out_dim, nb_train, weights1, weights2, beta);
		
		//Back-propagation phase
		backprop_batch(input, in_dim, hidden, delta_h, hid_dim, output, delta_o, targ, out_dim, nb_train, weights1, weights2, learn_rate, beta);
	
		for(i = 0; i < nb_train; i++)
			for(j = 0; j < out_dim; j++) 
				quad_error += (output[i][j] - targ[i][j])*(output[i][j] - targ[i][j]); 
		
		if (t%control_interv == 0)
			printf("Average training dataset quadratic error : %f \n", 0.5*quad_error/nb_train);
	}
	
	free(input[0]);
	free(input_test[0]);
	free(targ[0]);
	free(targ_test[0]);
	free(weights1[0]);
	free(weights2[0]);
	free(input);
	free(input_test);
	free(targ);
	free(targ_test);
	free(weights1);
	free(weights2);
	free(output);
	
	//######################## ##########################
	//######################## ##########################

	exit(EXIT_SUCCESS);	
}














