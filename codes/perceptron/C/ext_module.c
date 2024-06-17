

/* 
Neural networks pedagogical materials
The following code is free to use and modify to any extent (with no responsibility of the original author)

Reference to the author is a courtesy
Author : David Cornu => david.cornu@utinam.cnrs.fr
*/



#include "stdio.h"
#include "stdlib.h"


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



void forward(float *input, int in_dim, float *output, int out_dim, float **weights)
{
//######################## ##########################
//        One forward step with a binary neuron
//######################## ##########################

	float h;
	int i,j;

	for( i=0; i < out_dim; i++)
	{
		h = 0.0;
		for(j = 0; j < in_dim + 1; j++)
			h += weights[j][i]*input[j];
		
		if(h > 0)
			output[i] = 1.0;
		else
			output[i] = 0.0;
	}

}


void backprop(float *input, int in_dim, float *output, float *targ, int out_dim, float **weights, float learn_rate)
{
//######################## ##########################
//       One backward step with a binary neuron
//######################## ##########################
	
	int i, j;
	
	for(i = 0; i < in_dim + 1; i++)
		for(j = 0 ; j < out_dim; j++)
			weights[i][j] -= learn_rate*(output[j]-targ[j])*input[i];

}


void confmat(float **input, int in_dim, float **targ, int out_dim, int nb_data, float **weights)
{
//######################## ##########################
// Forward on an epoch and display a confusion matrix
//######################## ##########################
	
	float output[out_dim];
	int max_a, max_b;
	int confmatrix[out_dim][out_dim];
	float recall[out_dim], precis[out_dim];
	float accu;
	
	int i, j;
	
	accu = 0.0;
	for(i = 0; i < out_dim*out_dim; i++)
		confmatrix[0][i] = 0.0;
		
	for(i = 0; i < nb_data; i++)
	{
		forward(input[i], in_dim, output, out_dim, weights);
		
		max_a = argmax(output, out_dim);
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





































