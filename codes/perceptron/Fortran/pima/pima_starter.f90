

!############################### ################################
! perceptron exercise for the M2-OSAE Machine Learning lessons
! contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
!############################### ################################


module utils
	implicit none

contains

	subroutine shuffle(input, in_dim, targ, out_dim, nb_data)
		!######################## ##########################
		!               Fisher Yates Shuffle
		!######################## ##########################
		
		real, dimension(:,:), intent(INOUT) :: input, targ
		integer, intent(IN) :: in_dim, out_dim, nb_data
		
		real :: rdm
		integer :: i, ind
		real, allocatable :: temp_input(:), temp_targ(:)
		
		allocate(temp_input(in_dim))
		allocate(temp_targ(out_dim))
		
		do i=1, nb_data-1
			call random_number(rdm)
			ind = int(rdm * (nb_data-i)) + i
			
			temp_input(:) = input(i,:)
			input(i,:) = input(ind,:)
			input(ind,:) = temp_input(:)

			temp_targ(:) = targ(i,:)
			targ(i,:) = targ(ind,:)
			targ(ind,:) = temp_targ(:)

		end do
		
		deallocate(temp_input)
		deallocate(temp_targ)
	
	end subroutine !shuffle

end module utils


program logic_or
	
	use utils

	implicit none

	integer, parameter :: nb_dat = 768, in_dim = 8, out_dim = 2, nb_epochs = 1
	real, dimension(nb_dat,in_dim+1) :: input
	real, dimension(nb_dat,out_dim) :: targ
	real, dimension(out_dim) :: output
	real :: learn_rate = 0.1
	real, dimension(in_dim+1,out_dim) :: weights
	integer :: i, j, n, t, temp_class


	!######################## ##########################
	!          Loading data and pre process
	!######################## ##########################


	open(10, file="../../../data/pima-indians-diabetes.data")

	do i=1, nb_dat
		read(10,*) input(i,1:8), temp_class
		targ(i, temp_class+1) = 1.0
	end do
	input(:,9) = -1.0


	!######################## ##########################
	!          Initialize network weights
	!######################## ##########################

	call random_number(weights)
	weights(:,:) = weights(:,:)*(0.01)-0.005
	
	
	!######################## ##########################
	!                Main training loop
	!######################## ##########################
	!######################## ##########################
	
	

		!######################## ##########################
		!             Training on all data once
		!######################## ##########################
		
	
	!######################## ##########################
	!######################## ##########################
	
end program logic_or









