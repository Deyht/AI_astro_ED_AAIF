

!############################### ################################
! perceptron exercise for the M2-OSAE Machine Learning lessons
! contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
!############################### ################################


module utils
	implicit none

contains
	subroutine confmat(input, in_dim, targ, out_dim, nb_data, weights)
		!######################## ##########################
		! Forward on an epoch and display a confusion matrix
		!######################## ##########################
		
		real, dimension(:,:), intent(IN) :: input, targ
		integer, intent(IN) :: in_dim, out_dim, nb_data
		real, dimension(:,:), intent(IN) :: weights
		
		real, dimension(out_dim) :: output
		integer, dimension(1) :: max_a, max_b
		integer, dimension(out_dim,out_dim) :: confmatrix
		real, dimension(out_dim) :: recall, precis
		real :: accu, h
		
		integer :: i, j, n
		
		accu = 0
		confmatrix = 0
		do n=1, nb_data
		
			do i=1,out_dim
				h = 0.0
				do j=1, in_dim + 1
					h = h + weights(j,i)*input(n,j)
				end do
				
				if(h > 0) then
					output(i) = 1.0
				else
					output(i) = 0.0
				end if
			end do
			
			max_a = maxloc(output)
			max_b = maxloc(targ(n,:))
			confmatrix(max_b(1), max_a(1)) = confmatrix(max_b(1), max_a(1)) + 1
			if(max_a(1) == max_b(1)) then
				accu = accu + 1
			end if
		end do
		
		do i=1, out_dim
			recall(i) = 0
			precis(i) = 0
			do j=1, out_dim
				recall(i) = recall(i) + confmatrix(i,j)
				precis(i) = precis(i) + confmatrix(j,i)
			end do
			if(recall(i) > 0.0)then
				recall(i) = confmatrix(i,i) / recall(i) * 100.0
			end if
			if(precis(i) > 0.0)then
				precis(i) = confmatrix(i,i) /precis(i) * 100.0
			endif
		end do
			
		write(*,*) "*****************************************************************"
		write(*,'(a11,a44)') "Confmat :","Recall"
		do i=1, out_dim
			write(*,'(a10)', advance="no") "         "
			do j=1, out_dim
				write(*,'(I10)', advance="no") confmatrix(i,j)
			end do
			write(*,'(a12,f6.2)') "     ", recall(i)
		end do		
		write(*,*)
		write(*,'(a13)',advance="no") "Precision"
		do i=1, out_dim
			write(*,'(f10.2)', advance="no") precis(i)
		end do
		write(*,'(a7,f8.2)', advance="no") "Accu" , real(accu)/real(nb_data)*100.0
		write(*,*)
		write(*,*) "*****************************************************************"
		
	end subroutine !confmat

	
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

	integer, parameter :: nb_dat = 150, in_dim = 4, out_dim = 3, nb_epochs = 5000
	real, dimension(nb_dat,in_dim+1) :: input
	real, dimension(nb_dat,out_dim) :: targ
	real, dimension(out_dim) :: output
	real :: h, learn_rate = 0.005
	real, dimension(in_dim+1,out_dim) :: weights
	integer :: i, j, n, t, ind, temp_class
	


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
		do n=1, nb_dat
			!Forward phase
			do i=1,out_dim
				h = 0.0
				do j=1, in_dim + 1
					h = h + weights(j,i)*input(n,j)
				end do
				
				if(h > 0) then
					output(i) = 1.0
				else
					output(i) = 0.0
				end if
			end do
			
			!Back-propagation phase
			do i=1, in_dim + 1
				do j=1, out_dim
					weights(i,j) = weights(i,j) - learn_rate*(output(j)-targ(n,j))*input(n,i)
				end do			
			end do
			
		end do
	end do
	
	!######################## ##########################
	!######################## ##########################
	
end program logic_or














