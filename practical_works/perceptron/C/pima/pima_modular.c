


/* 
Neural networks pedagogical materials
The following code is free to use and modify to any extent (with no responsibility of the original author)

Reference to the author is a courtesy
Author : David Cornu => david.cornu@observatoiredeparis.psl.eu
*/



#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "tgmath.h"

#include "../ext_module.h"


int main()
{
	int nb_dat = 768, in_dim = 8, out_dim = 2, nb_epochs = 400;
	float **input, **targ, *output, **weights;
	float learn_rate = 0.05, rdm, mean, maxval;
	int i, j, t, ind, temp_class;
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
		for(i = 0; i < nb_dat; i++)
		{
			//Forward phase
			forward(input[i], in_dim, output, out_dim, weights);
			
			//Back-propagation phase
			backprop(input[i], in_dim, output, targ[i], out_dim, weights, learn_rate);
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














