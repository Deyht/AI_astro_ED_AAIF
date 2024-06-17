!############################### ################################
! K-means exercise for the M2-OSAE Machine Learning lessons
! contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
!############################### ################################


module utils
	implicit none

contains
	! This Function return a distance in ndim dimension
	! between two points (arguments are table with all the dimensions) 
	function dist(dat, cent, ndim)
		real :: dist
		real, intent(in) :: dat(:), cent(:)
		integer, intent(in) :: ndim
		integer :: i

		dist = 0.0
		do i = 1, ndim
			dist = dist + (dat(i)-cent(i))**2
		end do
	end function dist

end module utils



program kmeans

	use utils
	implicit none

	! Usefull data, feel free to add the ones you may need
	! for your own implementation of the algorithm
	real, allocatable :: input_data(:,:), centers(:,:), new_centers(:,:)
	integer, allocatable :: nb_points_per_center(:)
	integer :: nb_dim, nb_data
	real :: v_min, v_temp, cent_mov, all_dist, eps, rand
	integer :: nb_k, i_min
	integer :: i, j, l
	character (len=2) :: data_type
	character (len=60) :: file_name

	data_type = "2d"

	eps = 0.001
	nb_k = 4
	l = 0

	! This entry file must be edited to change to other
	! number of dimension. The code must be re-compiled !
	file_name = "kmeans_input_file_"//data_type//".dat"
	open(unit = 10, file = file_name)

	! Read the dimensions of the data in the file
	read(10, *) nb_dim, nb_data
	allocate(input_data(nb_data, nb_dim))

	write(*,*) nb_dim, nb_data

	! Load all the data
	do i = 1, nb_dim
		read(10, *) input_data(:, i)
	end do

	close(10)

	! Allocate the tables according to the dimension
	! gave in the input file
	allocate(centers(nb_k, nb_dim))
	allocate(new_centers(nb_k, nb_dim))
	allocate(nb_points_per_center(nb_k))

	! The origin of the centers are selected randomly
	! to the position of some points in the dataset
	do i=1, nb_k
		call random_number(rand)
		centers(i, :) = input_data(int(rand*nb_data),:)
	end do

	!############################### ################################
	!     Main loop, until the new centers do not move anymore
	!############################### ################################
	do
		l = l + 1
		
		all_dist = 0.0
		
		! Reset the working memory from the previous iteration
		do i = 1, nb_k
			new_centers(i,:) = 0.0
		end do
		nb_points_per_center(:) = 0

		!############################### ################################
		!         Association phase, loop on the data points
		!############################### ################################
		do i = 1, nb_data

			! Find the nearest point
			i_min = 1
			v_min = dist(input_data(i,:), centers(1,:), nb_dim)
			do j = 2, nb_k
				v_temp = dist(input_data(i,:), centers(j,:), nb_dim)
				if (v_temp <= v_min) then
					v_min = v_temp
					i_min = j
				end if
			end do
			
			all_dist = all_dist + v_min
			
			! Use in advance the new_centers vector for summing the positions
			new_centers(i_min,:) = new_centers(i_min,:) + input_data(i,:)
			! Update the number of points associated with this cluster center
			nb_points_per_center(i_min) = nb_points_per_center(i_min) + 1

		end do ! Data point loop

		!############################### ################################
		!           Update phase, calculate the new centers
		!############################### ################################
		do i = 1, nb_k
			if(nb_points_per_center(i) /= 0) then
				new_centers(i,:) = new_centers(i,:) / nb_points_per_center(i)
			end if
		end do


		! Calculate the sum of distances between the centers and the new ones
		cent_mov = 0.0
		do i = 1, nb_k
			cent_mov = cent_mov + dist(centers(i,:), new_centers(i,:), nb_dim)
		end do
		write(*,*) "Step :", l, " error :", all_dist/nb_data, " cent. move :", cent_mov


		! Effectivly move the centers by puting them at the centroids position
		do i = 1, nb_k
			if(nb_points_per_center(i) /= 0) then
				centers(i,:) = new_centers(i,:)
			end if
		end do


		! End the loop if the overall distance is less than a defined epsilon
		! or if too much iteration has been reached
		if(cent_mov <= eps .OR. l >= 100) then
			exit
		end if

	end do ! Main loop

	!############################### ################################
	!      Save the ending centroid position for visualization
	!############################### ################################
	file_name = "kmeans_output_"//data_type//".dat"
	open(unit = 10, file = file_name)
	
	write(10,*) nb_dim, nb_k
	
	do i = 1, nb_k
		write (10,*) centers(i, :)
	end do
	
	
	deallocate(input_data)
	deallocate(centers)
	deallocate(new_centers)
	deallocate(nb_points_per_center)
	

end program kmeans








