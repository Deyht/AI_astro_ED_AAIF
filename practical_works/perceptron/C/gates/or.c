
#include <stdio.h>
#include <stdlib.h>



int main()
{

	int input[4][3] = {0, 0, -1, 0, 1, -1, 1, 0, -1, 1, 1, -1}, targ[4] = {0, 1, 1, 1}, output;
	float h, learn_rate = 0.1f;
	float weights[3];
	int i, j, t;


	for(i = 0; i < 3; i++)
		weights[i] = (rand()/(float)RAND_MAX)*(0.02f)-0.01f;

	//######################## ##########################
	//                Main training loop
	//######################## ##########################
	for(t = 0; t < 5; t++)
	{
		printf("*** Iteration : %d ***\n", t);
		//######################## ##########################
		// Testing the result of the network with a forward
		//######################## ##########################

		for(i = 0; i < 4; i++)
		{
			// Forward phase
			h = 0.0f;
			for(j = 0; j < 3; j++)
				h = h + weights[j]*input[i][j];
			
			if(h > 0)
				output = 1;
			else
				output = 0;
		
			for(j = 0; j < 3; j++)
				printf("%d ", input[i][j]);
			printf("\nTarget : %d\n", targ[i]);
			printf("Output : %d\n\n", output);
		}


		//######################## ##########################
		//             Training on all data once
		//######################## ##########################
		for(i = 0; i < 4; i++)
		{
			//Forward Step
			h = 0.0f;
			for(j = 0; j < 3; j++)
				h = h + weights[j]*input[i][j];
			
			if(h > 0)
				output = 1;
			else
				output = 0;
			
			//Back-propagation phase
			for(j = 0; j < 3; j++)
				weights[j] = weights[j] - learn_rate*(output-targ[i])*input[i][j];
				
		}
	}

	exit(EXIT_SUCCESS);
}






