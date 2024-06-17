

! Neural networks pedagogical materials
! The following code is free to use and modify to any extent 
! (with no responsibility of the original author)

! Reference to the author is a courtesy
! Author : David Cornu => david.cornu@utinam.cnrs.fr



module ext_module
	
	implicit none
	
	contains
	
		subroutine forward(input, in_dim, output, out_dim, weights)
			!######################## ##########################
			!       One forward step with a binary neuron
			!######################## ##########################
			
			real, dimension(:), intent(IN) :: input
			real, dimension(:), intent(INOUT) :: output
			integer, intent(IN) :: in_dim, out_dim
			real, dimension(:,:), intent(IN) :: weights
		
			real :: h
			integer :: i,j
		
			do i=1,out_dim
				h = 0.0
				do j=1, in_dim + 1
					h = h + weights(j,i)*input(j)
				end do
				
				if(h > 0) then
					output(i) = 1.0
				else
					output(i) = 0.0
				end if
			end do
		
		end subroutine !forward
		
		
		subroutine backprop(input, in_dim, output, targ, out_dim, weights, learn_rate)
			!######################## ##########################
			!       One backward step with a binary neuron
			!######################## ##########################
		
			real, dimension(:), intent(IN) :: input, output, targ
			integer, intent(IN) :: in_dim, out_dim
			real, dimension(:,:), intent(INOUT) :: weights
			real, intent(IN) :: learn_rate
			
			integer :: i, j
			
			do i=1, in_dim + 1
				do j=1, out_dim
					weights(i,j) = weights(i,j) - learn_rate*(output(j)-targ(j))*input(i)
				end do			
			end do
		
		end subroutine !backprop
	
	
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
			real :: accu
			
			integer :: i, j
			
			accu = 0
			confmatrix = 0
			do i=1, nb_data
			
				call forward(input(i,:), in_dim, output, out_dim, weights)
				
				max_a = maxloc(output)
				max_b = maxloc(targ(i,:))
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
	



end module




