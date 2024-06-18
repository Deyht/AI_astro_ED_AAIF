

#include "stdio.h"
#include "stdlib.h"
#include "time.h"

#include "../ext_module.h"


int main()
{
	int nb_train = 568, nb_test=50, in_dim = 8, hid_dim = 8, out_dim = 2, nb_epochs = 1500, control_interv=100;
	float **input, **input_test, **targ, **targ_test, *hidden, *output, **weights1, **weights2;
	float h, learn_rate = 0.05, rdm, mean, maxval, beta=1.0, quad_error;
	int i, j, t, ind, temp_class;
	FILE *f = NULL;

	srand(time(NULL));

	input = create_2d_table(nb_train,in_dim+1);
	input_test = create_2d_table(nb_test,in_dim+1);
	targ = create_2d_table(nb_train, out_dim);
	targ_test = create_2d_table(nb_test, out_dim);
	hidden = (float*) calloc(hid_dim+1,sizeof(float));
	output = (float*) calloc(out_dim,sizeof(float));
	weights1 = create_2d_table(in_dim+1,hid_dim);
	weights2 = create_2d_table(hid_dim+1,out_dim);

	
	//######################## ##########################
	//          Loading data and pre process
	//######################## ##########################
	
	f = fopen("../../../data/pima-indians-diabetes.data", "r+");
	if(f == NULL)
	{
		printf("File not found\n");
		exit(EXIT_FAILURE);
	}
	
	for(i = 0; i < nb_train*(out_dim); i++)	
		targ[0][i] = 0.0;
	
	for(i = 0; i < nb_test*(out_dim); i++)	
		targ_test[0][i] = 0.0;
	
	for(i = 0; i < nb_train; i++)
	{
		for(j = 0; j < in_dim; j++)
			fscanf(f, "%f ", &input[i][j]);
		fscanf(f, "%d\n",  &temp_class);
		input[i][in_dim] = -1.0;
		targ[i][temp_class] = 1.0;
	}
	
	for(i = 0; i < nb_test; i++)
	{
		for(j = 0; j < in_dim; j++)
			fscanf(f, "%f ", &input_test[i][j]);
		fscanf(f, "%d\n",  &temp_class);
		input_test[i][in_dim] = -1.0;
		targ_test[i][temp_class] = 1.0;
	}
	
	for(i = 0; i < nb_train; i++)
	{
		if(input[i][0] > 8) input[i][0] = 8;
		input[i][7] = ((int)(input[i][7]-30))%10;
		if(input[i][7] > 5) input[i][7] = 5;
	}
	
	for(i = 0; i < nb_test; i++)
	{
		if(input_test[i][0] > 8) input_test[i][0] = 8;
		input_test[i][7] = ((int)(input_test[i][7]-30))%10;
		if(input_test[i][7] > 5) input_test[i][7] = 5;
	}
	
	fclose(f);

	for(i = 0; i < in_dim; i++)
	{
		mean = 0;
		for(j = 0; j < nb_train; j++)
			mean += input[j][i];
		for(j = 0; j < nb_test; j++)
			mean += input_test[j][i];
		mean /= (nb_train+nb_test);
		for(j = 0; j <  nb_train; j++)
			input[j][i] -= mean;
		for(j = 0; j <  nb_test; j++)
			input_test[j][i] -= mean;
		maxval = abs(input[0][i]);
		for(j = 0; j < nb_train; j++)
		{
			if(abs(input[j][i] > maxval))
				maxval = abs(input[j][i]);
		}
		for(j = 0; j < nb_test; j++)
		{
			if(abs(input_test[j][i] > maxval))
				maxval = abs(input_test[j][i]);
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
		for(j = 0; j < hid_dim; j++)
			weights1[i][j] = (rand()/(double)RAND_MAX)*(0.02)-0.01;
	for(i = 0; i < hid_dim+1; i++)
		for(j = 0; j < out_dim; j++)
			weights2[i][j] = (rand()/(double)RAND_MAX)*(0.02)-0.01;

	
	//######################## ##########################
	//                Main training loop
	//######################## ##########################
	//######################## ##########################
	
	for(t = 0; t < nb_epochs; t++)
	{
		if (t%control_interv == 0 || t == 0)
		{
			printf("\nIteration : %d\n", t);
		
			confmat(input_test, in_dim, hid_dim, targ_test, out_dim, nb_test, weights1, weights2, beta);
		}

		shuffle(input, in_dim+1, targ, out_dim, nb_train);

		quad_error = 0.0;
		//######################## ##########################
		//             Training on all data once
		//######################## ##########################
		for(i=0; i < nb_train; i++)
		{
			//Forward phase
			forward(input[i], in_dim, hidden, hid_dim, output, out_dim, weights1, weights2, beta);
			
			//Back-propagation phase
			backprop(input[i], in_dim, hidden, hid_dim, output, targ[i], out_dim, weights1, weights2, learn_rate, beta);
		
			for(j = 0; j < out_dim; j++) 
				quad_error += (output[j] - targ[i][j])*(output[j] - targ[i][j]); 
		}
		
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














