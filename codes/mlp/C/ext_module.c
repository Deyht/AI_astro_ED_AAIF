

/* 
Neural networks pedagogical materials
The following code is free to use and modify to any extent (with no responsibility of the original author)

Reference to the author is a courtesy
Author : David Cornu => david.cornu@utinam.cnrs.fr
*/


#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>

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



void forward(float *input, int in_dim, float* hidden, int hid_dim, float *output, int out_dim, float **weights1, float **weights2, float beta)
{
//######################## ##########################
//        One forward step with a logistic neuron
//######################## ##########################

	float h;
	int i,j;

	for(i = 0; i < hid_dim; i++)
	{
		h = 0.0;
		for(j = 0; j < in_dim + 1; j++)
			h += weights1[j][i]*input[j];
		hidden[i] = 1.0 / (1.0 + expf(-beta*h));
	}
	hidden[hid_dim] = -1.0;

	for(i = 0; i < out_dim; i++)
	{
		h = 0.0;
		for(j = 0; j < hid_dim + 1; j++)
			h += weights2[j][i]*hidden[j];
		output[i] = 1.0 / (1.0 + expf(-beta*h));
	}
	

}


void backprop(float *input, int in_dim, float *hidden, int hid_dim, float *output, float *targ, int out_dim, float **weights1, float **weights2, float learn_rate, float beta)
{
//######################## ##########################
//       One backward step with a logistic neuron
//######################## ##########################
	float delta_o[out_dim], delta_h[hid_dim+1];
	
	int i, j;
	float h;
	
	for(i = 0; i < out_dim; i++)
		delta_o[i] = beta*(output[i] - targ[i])* output[i] * (1.0 - output[i]);
	
	for(i = 0; i < hid_dim; i++)
	{
		h = 0.0;
		for(j = 0; j < out_dim; j++)
			h += weights2[i][j]*delta_o[j];
		delta_h[i] = beta*hidden[i]*(1.0 - hidden[i])*h;
	}
	
	for(i = 0; i < in_dim + 1; i++)
		for(j = 0 ; j < hid_dim; j++)
			weights1[i][j] -= learn_rate*delta_h[j]*input[i];
			
	for(i = 0; i < hid_dim + 1; i++)
		for(j = 0 ; j < out_dim; j++)
			weights2[i][j] -= learn_rate*delta_o[j]*hidden[i];
}


void confmat(float **input, int in_dim, int hid_dim, float **targ, int out_dim, int nb_data, float **weights1, float **weights2, float beta)
{
//######################## ##########################
// Forward on an epoch and display a confusion matrix
//######################## ##########################
	
	float output[out_dim];
	int max_a, max_b;
	int confmatrix[out_dim][out_dim];
	float recall[out_dim], precis[out_dim];
	float accu, quad_error;
	float hidden[hid_dim+1];
	
	int i, j;
	
	accu = 0.0;
	for(i = 0; i < out_dim*out_dim; i++)
		confmatrix[0][i] = 0.0;
		
	quad_error = 0.0;
	for(i = 0; i < nb_data; i++)
	{
		forward(input[i], in_dim, hidden, hid_dim, output, out_dim, weights1, weights2, beta);
		
		for(j = 0; j < out_dim; j++) 
			quad_error += (output[j] - targ[i][j])*(output[j] - targ[i][j]); 
		max_a = argmax(output, out_dim);
		max_b = argmax(targ[i], out_dim);
		confmatrix[max_b][max_a] += 1;
		if(max_a == max_b)
			accu = accu + 1;
	}
	
	for(i = 0; i < out_dim; i++)
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
	for(i = 0; i < out_dim; i++)
	{
		printf("         ");
		for(j = 0; j < out_dim; j++)
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





































