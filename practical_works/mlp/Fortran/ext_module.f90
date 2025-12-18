


! Neural networks pedagogical materials
! The following code is free to use and modify to any extent 
! (with no responsibility of the original author)

! Reference to the author is a courtesy
! Author : David Cornu => david.cornu@utinam.cnrs.fr




module ext_module
	
	implicit none
	
	contains	
	

	!######################## ########################## ##########################
	
	!    						 ON-LINE FUNCTIONS
	
	!######################## ########################## ##########################
	
		
		subroutine forward(input, in_dim, hidden, hid_dim, output, out_dim, weights1, weights2, beta)
			!######################## ##########################
			!    Forward the all netork with logistic neurons
			!######################## ##########################
			real, dimension(:), intent(IN) :: input
			real, dimension(:), intent(INOUT) :: hidden
			real, dimension(:), intent(INOUT) :: output
			real, intent(IN) :: beta
			integer, intent(IN) :: in_dim, hid_dim, out_dim
			real, dimension(:,:), intent(IN) :: weights1, weights2
			
			real :: h
			integer :: i,j
			
			do i=1, hid_dim
				h = 0.0
				do j=1, in_dim+1
					h = h + weights1(j,i)*input(j)
				end do
				hidden(i) = 1.0/(1.0 + exp(-beta*h))
			end do
			hidden(hid_dim+1) = -1.0

			do i=1, out_dim
				h = 0.0
				do j=1, hid_dim+1
					h = h + weights2(j,i)*hidden(j)
				end do
				output(i) = 1.0/(1.0 + exp(-beta*h))
			end do
		
		end subroutine !forward
		
			
		
		
		
		
		subroutine backprop(input, in_dim, hidden, hid_dim, output, targ, out_dim, weights1, weights2, learn_rate, beta)
			!######################## ##########################
			! Backward on the all network with logistic neurons
			!######################## ##########################
		
			real, dimension(:), intent(IN) :: input, hidden, output, targ
			integer, intent(IN) :: in_dim, hid_dim, out_dim
			real, dimension(:,:), intent(INOUT) :: weights1, weights2
			real, intent(IN) :: learn_rate, beta
			
			real, dimension(out_dim) :: delta_o
			real, dimension(hid_dim) :: delta_h
			integer :: i, j
			real :: h
		
			do i=1, out_dim
				delta_o(i) = beta*(output(i)-targ(i))*output(i)*(1.0-output(i))
			end do

			do i=1, hid_dim
				h = 0.0
				do j=1, out_dim
					h = h + weights2(i,j)*delta_o(j)
				end do
				delta_h(i) = beta*hidden(i)*(1.0-hidden(i))*h
			end do


			do i=1, in_dim+1
				do j=1, hid_dim
					weights1(i,j) = weights1(i,j) - learn_rate*(delta_h(j)*input(i))
				end do
			end do
			
			do i=1, hid_dim+1
				do j=1, out_dim
					weights2(i,j) = weights2(i,j) - learn_rate*(delta_o(j)*hidden(i))
				end do
			end do
		
		
		end subroutine !backward
		
		
	
	
	
		subroutine confmat(input, in_dim, hid_dim, targ, out_dim, nb_data, weights1, weights2, beta)
			!######################## ##########################
			! Forward on an epoch and display a confusion matrix
			!######################## ##########################
			
			real, dimension(:,:), intent(IN) :: input, targ
			integer, intent(IN) :: in_dim, hid_dim, out_dim, nb_data
			real, dimension(:,:), intent(IN) :: weights1, weights2
			real, intent(IN) :: beta
			
			real, dimension(out_dim) :: output
			real, dimension(hid_dim+1) :: hidden
			integer, dimension(1) :: max_a, max_b
			integer, dimension(out_dim,out_dim) :: confmatrix
			real, dimension(out_dim) :: recall, precis
			real :: accu, quad_error
			
			integer :: i, j
			
			accu = 0
			confmatrix = 0
			quad_error = 0
			do i=1, nb_data
			
				call forward(input(i,:), in_dim, hidden, hid_dim, output, out_dim, weights1, weights2, beta)
				
				quad_error = quad_error + 0.5*sum((output(:) - targ(i,:))**2)
				
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
			write(*,*) "Average test set quadratic error : ", quad_error/real(nb_data)
			
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
		
		
		
		
	!######################## ########################## ##########################
	
	!    						   BATCH FUNCTIONS
	
	!######################## ########################## ##########################
	
	


		subroutine forward_batch(input, in_dim, hidden, hid_dim, output, out_dim, weights1, weights2, beta)
			!######################## ##########################
			!    Forward the all netork with logistic neurons
			!######################## ##########################
			real, dimension(:,:), intent(IN) :: input
			real, dimension(:,:), intent(INOUT) :: hidden
			real, dimension(:,:), intent(INOUT) :: output
			real, intent(IN) :: beta
			integer, intent(IN) :: in_dim, hid_dim, out_dim
			real, dimension(:,:), intent(IN) :: weights1, weights2
			
			real :: h
			integer :: i,j
			
			hidden = matmul(input, weights1)
			
			hidden(:,1:hid_dim) = 1.0/(1.0 + exp(-beta*hidden(:,1:hid_dim)))
			
			output = matmul(hidden, weights2)

			output(:,:) = 1.0/(1.0 + exp(-beta*output(:,:)))
		
		end subroutine !forward
		


		subroutine backprop_batch(input, in_dim, hidden, delta_h, hid_dim, output, &
			delta_o, targ, out_dim, weights1, weights2, learn_rate, beta)
			!######################## ##########################
			! Backward on the all network with logistic neurons
			!######################## ##########################
		
			real, dimension(:,:), intent(INOUT) :: input, hidden, delta_h, output, delta_o, targ
			integer, intent(IN) :: in_dim, hid_dim, out_dim
			real, dimension(:,:), intent(INOUT) :: weights1, weights2
			real, intent(IN) :: learn_rate, beta
			
			integer :: i, j
			real :: h
		
			delta_o(:,:) = beta*(output(:,:)-targ(:,:))*output(:,:)*(1.0-output(:,:))


			delta_h = matmul(delta_o, transpose(weights2))

			delta_h(:,1:hid_dim) = beta*hidden(:,1:hid_dim)*(1.0-hidden(:,1:hid_dim))*delta_h(:,1:hid_dim) 
			delta_h(:,hid_dim+1) = 0.0


			weights2 = weights2 - learn_rate*matmul(transpose(hidden),delta_o)
			weights1 = weights1 - learn_rate*matmul(transpose(input),delta_h)
		
		end subroutine !backward




		subroutine confmat_batch(input, in_dim, hidden, hid_dim, output, targ, out_dim, nb_data, weights1, weights2, beta)
			!######################## ##########################
			! Forward on an epoch and display a confusion matrix
			!######################## ##########################
			
			real, dimension(:,:), intent(INOUT) :: input, output, targ, hidden
			integer, intent(IN) :: in_dim, hid_dim, out_dim, nb_data
			real, dimension(:,:), intent(IN) :: weights1, weights2
			real, intent(IN) :: beta
			
			real, dimension(out_dim) :: temp_targ
			integer, dimension(1) :: max_a, max_b
			integer, dimension(out_dim,out_dim) :: confmatrix
			real, dimension(out_dim) :: recall, precis
			real :: accu, quad_error
			
			integer :: i, j
			
			accu = 0
			confmatrix = 0
			quad_error = 0

			call forward_batch(input, in_dim, hidden, hid_dim, output, out_dim, weights1, weights2, beta)
				
			quad_error = quad_error + 0.5*sum((output(:,:) - targ(:,:))**2)
				
			do i = 1, nb_data
				max_a = maxloc(output(i,:))
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
			write(*,*) "Average test set quadratic error: ", quad_error/real(nb_data)
			
		end subroutine !confmat






end module




