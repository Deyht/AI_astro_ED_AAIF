
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



int main()
{
	int nb_dat = 768, in_dim = 8, out_dim = 2, nb_epochs = 1;
	float **input, **targ, *output, **weights;
	float learn_rate = 0.1;
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
	
	

		//######################## ##########################
		//             Training on all data once
		//######################## ##########################
		
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














