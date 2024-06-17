


! Neural networks pedagogical materials
! The following code is free to use and modify to any extent 
! (with no responsibility of the original author)

! Reference to the author is a courtesy
! Author : David Cornu => david.cornu@observatoiredeparis.psl.eu


program logic_or
	
	use ext_module

	implicit none

	integer, parameter :: nb_dat = 150, in_dim = 4, out_dim = 3, nb_epochs = 5000
	real, dimension(nb_dat,in_dim+1) :: input
	real, dimension(nb_dat,out_dim) :: targ
	real, dimension(out_dim) :: output
	real :: h, learn_rate = 0.005
	real, dimension(in_dim+1,out_dim) :: weights
	integer :: i, t, ind, temp_class
	


	!######################## ##########################
	!          Loading data and pre process
	!######################## ##########################
	
	open(10, file="../../../data/iris.data")
	
	targ(:,:) = 0
	do i=1, nb_dat
		read(10,*) input(i,1:4), temp_class
		targ(i, temp_class+1) = 1.0
	end do
	input(:,5) = -1.0

	do i=1, 4
		input(:,i) = input(:,i)-(sum(input(:,i))/nb_dat)
		input(:,i) = input(:,i)/maxval(abs(input(:,i)))
	end do


	
	!######################## ##########################
	!          Initialize network weights
	!######################## ##########################

	call random_number(weights)
	weights(:,:) = weights(:,:)*(0.02)-0.01
	
	
	!######################## ##########################
	!                Main training loop
	!######################## ##########################
	!######################## ##########################
	
	do t = 1, nb_epochs
	
		if (mod(t,200) == 0 .OR. t == 1) then
			Write(*,*)
			write(*,*) "Iteration :", t
		
			call confmat(input(:,:), in_dim, targ(:,:), out_dim, nb_dat, weights)
		end if

		call shuffle(input, in_dim+1, targ, out_dim, nb_dat)

		!######################## ##########################
		!             Training on all data once
		!######################## ##########################
		do i=1, nb_dat
			!Forward phase
			call forward(input(i,:), in_dim, output, out_dim, weights)
			
			!Back-propagation phase
			call backprop(input(i,:), in_dim, output, targ(i,:), out_dim, weights, learn_rate)
		end do
	end do
	
	!######################## ##########################
	!######################## ##########################
	
end program logic_or














