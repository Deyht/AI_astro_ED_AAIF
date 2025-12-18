

/* 
Neural networks pedagogical materials
The following code is free to use and modify to any extent (with no responsibility of the original author)

Reference to the author is a courtesy
Author : David Cornu => david.cornu@utinam.cnrs.fr
*/


float **create_2d_table(int m, int n);
int argmax(float *tab, int size);
void forward(float *input, int in_dim, float* hidden, int hid_dim, float *output, int out_dim, float **weights1, float **weights2, float beta);
void backprop(float *input, int in_dim, float *hidden, int hid_dim, float *output, float *targ, int out_dim, float **weights1, float **weights2, float learn_rate, float beta);
void confmat(float **input, int in_dim, int hid_dim, float **targ, int out_dim, int nb_data, float **weights1, float **weights2, float beta);
void shuffle(float** input, int in_dim, float** targ, int out_dim, int nb_data);
