program logic_or
	implicit none

	integer, dimension(4,3) :: input
	integer, dimension(4) :: targ
	integer :: output
	real :: h, learn_rate = 0.1
	real, dimension(3) :: weights
	integer :: i, j, t 

	input = reshape((/0, 0, 1, 1, 0 ,1 ,0 ,1, -1, -1, -1, -1 /), shape(input))
	targ = (/0, 1, 1, 1/)

	call random_number(weights)
	weights(:) = weights(:)*(0.02)-0.01

	!######################## ##########################
	!                Main training loop
	!######################## ##########################
	do t = 1, 5
		write(*,*) "Iteration :", t
		!######################## ##########################
		! Testing the result of the network with a forward
		!######################## ##########################

		do i=1, 4
			!Forward phase
			h = 0.0
			do j=1, 3
				h = h + weights(j)*input(i,j)
			end do
			
			if(h > 0) then
				output = 1
			else
				output = 0
			end if
		
			write(*,*) "Input  :", input(i,:)
			write(*,*) "Target :", targ(i)
			write(*,*) "Output :", output
			write(*,*) " "
		end do


		!######################## ##########################
		!             Training on all data once
		!######################## ##########################
		do i=1, 4
			!Forward Step
			h = 0.0
			do j=1, 3
				h = h + weights(j)*input(i,j)
			end do
			
			if(h > 0) then
				output = 1
			else
				output = 0
			end if
			
			!Back-propagation phase
			do j=1, 3
				weights(j) = weights(j) - learn_rate*(output-targ(i))*input(i,j)
			end do
		end do
	end do
end program logic_or

