

! Neural networks pedagogical materials
! The following code is free to use and modify to any extent 
! (with no responsibility of the original author)

! Reference to the author is a courtesy
! Author : David Cornu => david.cornu@utinam.cnrs.fr


program multi_layer_perceptron
	
	use ext_module
	
	implicit none

	integer, parameter :: nb_train = 915, nb_test = 200, in_dim = 3753, hid_dim = 8
	integer, parameter :: out_dim = 7, nb_epochs = 100, control_interv = 5
	real, dimension(nb_train,in_dim+1) :: input
	real, dimension(nb_train,out_dim) :: targ
	real, dimension(nb_test,in_dim+1) :: input_test
	real, dimension(nb_test,out_dim) :: targ_test
	real, dimension(out_dim) :: output
	real, dimension(hid_dim+1) :: hidden
	real, dimension(hid_dim) :: delta_h
	real :: h, learn_rate = 0.2, beta = 1.0
	real, dimension(in_dim+1,hid_dim) :: weights1
	real, dimension(hid_dim+1,out_dim) :: weights2
	real, dimension(in_dim) :: read_buffer
	
	integer :: i, j, t, temp_class
	real, dimension(in_dim) :: means, max_val
	real :: quad_error, max_train, max_test


	!######################## ##########################
	!          Loading data and pre process
	!######################## ##########################
	
	!Instructions on how to download the data are provided in the python version of this script
	!1115 rows, spectra have 3753 pixels, and there is 7 output classes 
	open(10, file="stellar_spectra_data/train.dat")
	open(11, file="stellar_spectra_data/target.dat")
	! Warning : Elements in file must be shuffled before
	! spliting into train and test datasets
	
	! Load the training dataset part
	targ(:,:) = 0
	do i=1, nb_train
		read(10,*) read_buffer(1:in_dim)
		input(i,1:in_dim) = read_buffer(1:in_dim)
	end do
	input(:,in_dim+1) = -1.0
	
	do i=1, nb_train
		read(11,*) targ(i,1:out_dim)
	end do
	
	! Load the test dataset part
	targ_test(:,:) = 0
	do i=1, nb_test
		read(10,*) read_buffer(1:in_dim)
		input_test(i,1:in_dim) = read_buffer(1:in_dim)
	end do
	input_test(:,in_dim+1) = -1.0
	
	do i=1, nb_test
		read(11,*) targ_test(i,1:out_dim)
	end do

	! Warning : Train and test must be normalized exactly the same way
	do i = 1, in_dim
		means(i) = sum(input(:,i))
		means(i) = means(i) + sum(input_test(:,i))
		means(i) = means(i) / (nb_train + nb_test)
		input(:,i) = input(:,i)-means(i)
		input_test(:,i) = input_test(:,i) - means(i)
		
		max_train = maxval(abs(input(:,i)))
		max_test = maxval(abs(input_test(:,i)))
		if(max_test > max_train) then
			max_val(i) = max_test
		else
			max_val(i) = max_train
		end if
		input(:,i) = input(:,i)/max_val(i)
		input_test(:,i) = input_test(:,i)/max_val(i)
	end do


	!######################## ##########################
	!          Initialize network weights
	!######################## ##########################
	
	call random_number(weights1)
	call random_number(weights2)
	weights1(:,:) = weights1(:,:)*(0.02)-0.01
	weights2(:,:) = weights2(:,:)*(0.02)-0.01
	

	!######################## ##########################
	!                Main training loop
	!######################## ##########################
	do t = 1, nb_epochs
		if (mod(t,control_interv) == 0 .OR. t == 1) then
			write(*,*)
			write(*,*) "#########################################################################"
			write(*,*) "Iteration :", t
			! Testing the result of the network with a forward
			! and printing it in the form of a confusion matrix
			call confmat(input_test, in_dim, hid_dim, targ_test, out_dim, nb_test, weights1, weights2, beta)
		end if
		
		
		call shuffle(input, in_dim+1, targ, out_dim, nb_train)

		!######################## ##########################
		!             Training on all data once
		!######################## ##########################
		quad_error = 0.0
		do i=1, nb_train

			call forward(input(i,:), in_dim, hidden, hid_dim, output, out_dim, weights1, weights2, beta)
			
			call backprop(input(i,:), in_dim, hidden, hid_dim, output, targ(i,:), out_dim, weights1, weights2, learn_rate, beta)
			
			! Compute error on training dataset
			quad_error = quad_error + 0.5*sum((output(:) - targ(i,:))**2)
		end do
		
		if (mod(t,control_interv) == 0) then
			write(*,*)
			write(*,*) "Average training dataset quadratic error :", quad_error/nb_train	
		end if

	end do
	
end program multi_layer_perceptron






