
//############################### ################################
// perceptron exercise for the M2-OSAE Machine Learning lessons
// contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
//############################### ################################

#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "tgmath.h"


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



void shuffle(float** input, int in_dim, float** targ, int out_dim, int nb_data)
{
//######################## ##########################
//               Fisher Yates Shuffle
//######################## ##########################
	
	int i, j, ind;
	float temp;
	
	for(i = 0; i < nb_data-1 ; i++)
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


void confmat(float **input, int in_dim, float **targ, int out_dim, int nb_data, float **weights)
{
//######################## ##########################
// Forward on an epoch and display a confusion matrix
//######################## ##########################
	
	float output[out_dim];
	int max_a, max_b;
	int confmatrix[out_dim][out_dim];
	float recall[out_dim], precis[out_dim];
	float accu, h;
	
	int i, j, n;
	
	accu = 0.0;
	for(i = 0; i < out_dim*out_dim; i++)
		confmatrix[0][i] = 0.0;
		
	for(n = 0; n < nb_data; n++)
	{

		for(i = 0; i < out_dim; i++)
		{
			h = 0.0;
			for(j = 0; j < in_dim + 1; j++)
				h += weights[j][i]*input[n][j];
			
			if(h > 0)
				output[i] = 1.0;
			else
				output[i] = 0.0;
		}
		
		max_a = argmax(output, out_dim);
		max_b = argmax(targ[n], out_dim);
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
	for(i = 0; i < out_dim; i++)
		printf("%10.2f", precis[i]);

	printf(" Accu  %6.2f" , ((float)accu/(float)nb_data)*100.0);
	printf("\n*****************************************************************\n");
	
}


int main()
{
	int nb_dat = 768, in_dim = 8, out_dim = 2, nb_epochs = 400;
	float **input, **targ, *output, **weights;
	float h, learn_rate = 0.05, rdm, mean, maxval;
	int i, j, t, n, ind, temp_class;
	FILE *f = NULL;

	srand(time(NULL));

	input = create_2d_table(nb_dat,in_dim+1);
	targ = create_2d_table(nb_dat, out_dim);
	output = (float*) calloc(out_dim,sizeof(float));
	weights = create_2d_table(in_dim+1,out_dim);

	
	//######################## ##########################
	//          Loading data and pre process
	//######################## ##########################
	
	f = fopen("../../../data/pima-indians-diabetes.data", "r+");
	if(f == NULL)
	{
		printf("File not found\n");
		exit(EXIT_FAILURE);
	}
	
	for(i = 0; i < nb_dat*(out_dim); i++)	
		targ[0][i] = 0.0;
	
	for(i = 0; i < nb_dat; i++)
	{
		for(j = 0; j < in_dim; j++)
			fscanf(f, "%f ", &input[i][j]);
		fscanf(f, "%d\n",  &temp_class);
		input[i][in_dim] = -1.0;
		targ[i][temp_class] = 1.0;
	}
	
	fclose(f);
	
	for(i = 0; i < nb_dat; i++)
	{
		if(input[i][0] > 8) input[i][0] = 8;
		input[i][7] = ((int)(input[i][7]-30))%10;
		if(input[i][7] > 5) input[i][7] = 5;
	}

	for(i = 0; i < in_dim; i++)
	{
		mean = 0.0f;
		for(j = 0; j < nb_dat; j++)
			mean += input[j][i];
		mean /= nb_dat;
		for(j = 0; j <  nb_dat; j++)
			input[j][i] -= mean;
		maxval = fabsf(input[0][i]);
		for(j = 0; j < nb_dat; j++)
		{
			if(abs(input[j][i] > maxval))
				maxval = fabsf(input[j][i]);
		}
		for(j = 0; j < nb_dat; j++)
			input[j][i] /= maxval;
	}

	
	//######################## ##########################
	//          Initialize network weights
	//######################## ##########################

	for(i = 0; i < in_dim+1; i++)
		for(j = 0; j < out_dim; j++)
			weights[i][j] = (rand()/(double)RAND_MAX)*(0.01)-0.005;

	
	//######################## ##########################
	//                Main training loop
	//######################## ##########################
	//######################## ##########################
	
	for(t = 0; t < nb_epochs; t++)
	{
		if (t%10 == 0 || t == 0)
		{
			printf("\nIteration : %d\n", t);
		
			confmat(input, in_dim, targ, out_dim, nb_dat, weights);
		}

		shuffle(input, in_dim+1, targ, out_dim, nb_dat);

		//######################## ##########################
		//             Training on all data once
		//######################## ##########################
		for(n = 0; n < nb_dat; n++)
		{
			//Forward phase
			for(i = 0; i < out_dim; i++)
			{
				h = 0.0;
				for(j = 0; j < in_dim + 1; j++)
					h += weights[j][i]*input[n][j];
				
				if(h > 0)
					output[i] = 1.0;
				else
					output[i] = 0.0;
			}
			
			//Back-propagation phase
			for(i = 0; i < in_dim + 1; i++)
				for(j = 0 ; j < out_dim; j++)
					weights[i][j] -= learn_rate*(output[j]-targ[n][j])*input[n][i];
		}
	}
	
	free(input[0]);
	free(targ[0]);
	free(weights[0]);
	free(input);
	free(targ);
	free(weights);
	free(output);
	
	//######################## ##########################
	//######################## ##########################

	exit(EXIT_SUCCESS);	
}














