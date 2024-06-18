

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
#include <cblas.h>

float **create_2d_table(int m, int n)
{
	int i;
	float** tab;
	float* temp_tab;
	
	tab = (float**) malloc(m*sizeof(float*));
	temp_tab = (float*) malloc(m*n*sizeof(float));
	
	for(i = 0; i < m; i++)
		tab[i] = &temp_tab[i*n];
		
	return tab;
}



int argmax(float *tab, int size)
{
	int i;
	float max;
	int imax;

	max = *tab;
	imax = 0;
	
	
	for(i = 1; i < size; i++)
	{
		if(tab[i] >= max)
		{
			max = tab[i];
			imax = i;
		}
	}
	
	return imax;
}



void forward_batch(float **input, int in_dim, float** hidden, int hid_dim, float **output, int out_dim, int nb_dat, float **weights1, float **weights2, float beta)
{
//######################## ##########################
//        One forward step with a logistic neuron
//######################## ##########################

	float h;
	int i,j;
	
	
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nb_dat, hid_dim+1, in_dim+1, 1.0, *input, in_dim+1, *weights1, hid_dim+1, 0.0, *hidden, hid_dim+1);
	
	for(i = 0; i < nb_dat; i++)
	{
		for(j = 0; j < hid_dim; j++)
			hidden[i][j] = 1.0/(1.0 + exp(-beta*hidden[i][j]));
		hidden[i][hid_dim] = -1.0;
	}	
	
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nb_dat, out_dim, hid_dim+1, 1.0, *hidden, hid_dim+1, *weights2, out_dim, 0.0, *output, out_dim);


	for(i = 0; i < nb_dat; i++)
		for(j = 0; j < out_dim; j++)
			output[i][j] = 1.0/(1.0 + exp(-beta*output[i][j]));
}


void backprop_batch(float **input, int in_dim, float **hidden, float **delta_h, int hid_dim, float **output, float **delta_o, float **targ, int out_dim, int nb_dat, float **weights1, float **weights2, float learn_rate, float beta)
{
//######################## ##########################
//       One backward step with a logistic neuron
//######################## ##########################
	
	int i, j;
	float h;
	
	for(i = 0; i < nb_dat; i++)
		for(j = 0; j < out_dim; j++)
			delta_o[i][j] = beta*(output[i][j] - targ[i][j])* output[i][j] * (1.0 - output[i][j]);
	
	
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nb_dat, hid_dim+1, out_dim, 1.0, *delta_o, out_dim, *weights2, out_dim, 0.0, *delta_h, hid_dim+1);
	
	for(i = 0; i < nb_dat; i++)
	{
		for(j = 0; j < hid_dim; j++)
			delta_h[i][j] = beta*hidden[i][j]*(1.0-hidden[i][j])*delta_h[i][j] ;
		delta_h[i][hid_dim] = 0.0;
	}

	
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, hid_dim+1, out_dim, nb_dat, -learn_rate, *hidden, hid_dim+1, *delta_o, out_dim, 1.0, *weights2, out_dim);
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, in_dim+1, hid_dim+1, nb_dat, -learn_rate, *input, in_dim+1, *delta_h, hid_dim+1, 1.0, *weights1, hid_dim+1);
	
}


void confmat_batch(float **input, int in_dim, float **hidden, int hid_dim, float **output, float **targ, int out_dim, int nb_data, float **weights1, float **weights2, float beta)
{
//######################## ##########################
// Forward on an epoch and display a confusion matrix
//######################## ##########################
	
	int max_a, max_b;
	int confmatrix[out_dim][out_dim];
	float recall[out_dim], precis[out_dim];
	float accu, quad_error;
	
	int i, j;
	
	accu = 0.0;
	for(i = 0; i < out_dim*out_dim; i++)
		confmatrix[0][i] = 0.0;
		
	quad_error = 0.0;
	
	forward_batch(input, in_dim, hidden, hid_dim, output, out_dim, nb_data, weights1, weights2, beta);
	
	for(i=0; i < nb_data; i++)
	{
		for(j = 0; j < out_dim; j++) 
			quad_error += (output[i][j] - targ[i][j])*(output[i][j] - targ[i][j]); 
		max_a = argmax(output[i], out_dim);
		max_b = argmax(targ[i], out_dim);
		confmatrix[max_b][max_a] += 1;
		if(max_a == max_b)
			accu = accu + 1;
	}
	
	for(i=0; i < out_dim; i++)
	{
		recall[i] = 0;
		precis[i] = 0;
		for(j = 0; j < out_dim; j++)
		{
			recall[i] += confmatrix[i][j];
			precis[i] += confmatrix[j][i];
		}
		if(recall[i] > 0.0)
			recall[i] = confmatrix[i][i] / recall[i] * 100.0;

		if(precis[i] > 0.0)
			precis[i] = confmatrix[i][i] /precis[i] * 100.0;

	}
		
	printf("*****************************************************************\n");
	printf("Confmat :                                           Recall\n");
	for(i=0; i < out_dim; i++)
	{
		printf("         ");
		for(j=0; j < out_dim; j++)
			printf("%10d", confmatrix[i][j]);
		
		printf("        %6.2f\n", recall[i]);
	}	
	printf("\n  Precision");
	for(i=0; i < out_dim; i++)
		printf("%10.2f", precis[i]);

	printf(" Accu  %6.2f" , ((float)accu/(float)nb_data)*100.0);
	printf("\n*****************************************************************\n");
	printf("Average test set quadratic error: %f\n", quad_error/(float)nb_data);
}



void shuffle(float** input, int in_dim, float** targ, int out_dim, int nb_data)
{
//######################## ##########################
//               Fisher Yates Shuffle
//######################## ##########################
	
	int i, j, ind;
	float temp;
	
	for(i=0; i < nb_data-1 ; i++)
	{	
		
		ind = (int)(((double)rand()/(double)RAND_MAX) * (nb_data-i)) + i;
		
		for(j = 0; j < in_dim; j++)
		{
			temp = input[i][j];
			input[i][j] = input[ind][j];
			input[ind][j] = temp;
		}
		
		for(j = 0; j < out_dim; j++)
		{
			temp = targ[i][j];
			targ[i][j] = targ[ind][j];
			targ[ind][j] = temp;
		}
	}

}





































