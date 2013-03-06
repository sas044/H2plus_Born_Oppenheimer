module propagator
    !Propagates a wavefunction, using ode.f90 ("shago"), an algorithm by 
    !Shampine & Gordon, implemented by Burkhardt, (see comments in ode.f90).
    !The function input to ode makes up most of this program.

    !======================  Declarations  ===================================
    use hdf5
    
    use mpi
    implicit none
    
    !include 'mpif.h'

    !Parallelization variables. 
    integer					    :: nr_procs
    integer					    :: my_id
    !The electronic states that is the work of this processor.
    integer, allocatable, dimension(:)		    :: el_indices

    !Overlap matrix variables. 
    real (kind = 8), allocatable, dimension(:,:)    :: overlap_real
    complex (kind = 8), allocatable, dimension(:,:) :: overlap
    integer, allocatable, dimension(:)		    :: pivoting
    logical					    :: overlap_set = .false.

    !Hamiltonian matrix variables.    
    real (kind = 8), allocatable, dimension(:,:,:)  :: ti_hamiltonian_real
    real (kind = 8), allocatable, dimension(:,:)    :: td_hamiltonian_real
    complex (kind = 8), allocatable, dimension(:,:,:)  :: ti_hamiltonian
    complex (kind = 8), allocatable, dimension(:,:) :: td_hamiltonian
    integer					    :: nr_el, nr_vib, order
    integer					    :: nr_total, nr_total_el
    
    !Wavefunction.
    real (kind = 8), allocatable, dimension(:)	    :: y
    
    !Laser variables.
    real (kind = 8), parameter			    :: pi = 3.1415927
    real (kind = 8)				    :: amplitude, omega
    real (kind = 8)				    :: pulse_duration
    real (kind = 8)				    :: total_duration
    integer					    :: cycles

    !Filenames.
    character(len = 150)				    :: name_ti, name_td

    
    contains
    !=========================  Subroutines  =================================

    !--------------------  Setting the variables  ----------------------------
    
    subroutine initialize_mpi()
	implicit none
	integer			::  error_flag
	
	!Call MPI routines
	call MPI_INIT(error_flag)
	call MPI_COMM_RANK(MPI_COMM_WORLD, my_id, error_flag)
	call MPI_COMM_SIZE(MPI_COMM_WORLD, nr_procs, error_flag)

    end subroutine initialize_mpi

    subroutine finalize_mpi()
	implicit none
	integer			::  error_flag

	!Call MPI routines
	call mpi_barrier(MPI_COMM_WORLD, error_flag)
	call MPI_FINALIZE(error_flag)

    end subroutine finalize_mpi

    subroutine set_overlap()
	implicit none

	integer(HID_T)			    :: file_id, dset_id, dspace_id
	integer				    :: error, i
	integer(HSIZE_T), dimension(2)	    :: my_size, max_size
	!HDF5 stuff
	!==========
	!Open stuff.
	call h5open_f(error)
	call h5fopen_f(name_ti, H5F_ACC_RDONLY_F, file_id, error)
	call h5dopen_f(file_id, "/overlap", dset_id, error)
	call h5dget_space_f(dset_id, dspace_id, error)
	!---------------------------------------------

	!Get size.
	call h5sget_simple_extent_dims_f(dspace_id, my_size, max_size, error)
	
	nr_vib = my_size(1)

	!Set the overlap matrix to that size.
	allocate(overlap(nr_vib, nr_vib))
	allocate(overlap_real(nr_vib, nr_vib))
	allocate(pivoting(nr_vib))

	!Read out the overlap file.
	call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, overlap_real, my_size, error)
	!---------------------------------------------
	!Close stuff.
	call h5dclose_f(dset_id, error)
	call h5fclose_f(file_id, error)
	call h5close_f(error)
	!========
	

	!Read out the order.
	order = 0
	
	do i = 1, nr_vib

	    if(abs(overlap_real(1,i)) > 1e-13) then
		order = order + 1
	    end if
	end do
	
	!DEBUG
	if(my_id == 0) then
	    write(*,*) "The order is:", order
	end if
	!----

	!Convert to complex.
	overlap = overlap_real + complex(0,0)

	!Call LAPACK factorization routine.
	call zgetrf(nr_vib, nr_vib, overlap, nr_vib,  pivoting, error)

	if (error == 0) then
	    overlap_set = .true.
	else
	    write(*,*) "Factorization was not successful."
	end if


    end subroutine set_overlap


    subroutine set_ti_hamiltonian()
	implicit none
	
	integer(HID_T)			    :: file_id, dset_id, dspace_id, memspace_id
	integer				    :: error
	integer(HSIZE_T), dimension(3)	    :: my_size, max_size, sub_slice
	integer(HSIZE_T), dimension(3)	    :: count
	integer(HSIZE_T), dimension(3)	    :: offset

	!Open stuff.
	call h5open_f(error)
	call h5fopen_f(name_ti, H5F_ACC_RDONLY_F, file_id, error)
	call h5dopen_f(file_id, "/hamiltonian", dset_id, error)
	call h5dget_space_f(dset_id, dspace_id, error)
	!---------------------------------------------

	!Get size.
	call h5sget_simple_extent_dims_f(dspace_id, my_size, max_size, error)
	
	!Set number of states.
	nr_total_el = my_size(1)
	nr_total = nr_total_el * nr_vib

	!Allocate wavefunction.
	allocate(y(2*nr_total))
	
	!Distributing el-states among the processors.
	call distribute_work()

	!Set sub_slice size.
	sub_slice(1) = nr_el
	sub_slice(2) = my_size(2)
	sub_slice(3) = my_size(3) 

	!Allocate my_slice.
	allocate(ti_hamiltonian_real(sub_slice(1), sub_slice(2), sub_slice(3)))
	allocate(ti_hamiltonian(sub_slice(1), sub_slice(2), sub_slice(3)))

	!Create memory dataspace, shaped as the piece to be cut out.
	call h5screate_simple_f(3, sub_slice, memspace_id, error)

	!Hyperslab specs:
	!Starting point of the data rectangle.
	offset(1) = el_indices(1) - 1
	offset(2) = 0
	offset(3) = 0

	!Lengths of data rectangle.
	count(1)  = nr_el
	count(2)  = sub_slice(2)
	count(3)  = sub_slice(3)

	call h5sselect_hyperslab_f(dspace_id, H5S_SELECT_SET_F, offset, count, error)
	
	call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, ti_hamiltonian_real, sub_slice, error,&
	      memspace_id, dspace_id)

	!---------------------------------------------
	!Close stuff.
	call h5dclose_f(dset_id, error)
	call h5fclose_f(file_id, error)
	call h5close_f(error)
	
	ti_hamiltonian = ti_hamiltonian_real + complex(0,0)

    end subroutine set_ti_hamiltonian


    subroutine distribute_work()
	implicit none

	integer		    :: i
	
	!Number of electronic states for THIS processor.
	nr_el = nr_total_el / nr_procs
	
	!Assert equal distribution.
	if(nr_el * nr_procs == nr_total_el) then
	    
	    allocate(el_indices(nr_el))
	    
	    !Indices of this processor's el-states.
	    do i = 1, nr_el
		el_indices(i) = nr_el * (my_id) + i
	    end do

	else
	    write(*,*) "TROUBLE! Number of el-states is not dividable by"
	    write(*,*) "the number of processors!"
	    write(*,*) "nr_procs:", nr_procs
	    write(*,*) "nr_total_el:", nr_total_el
	    write(*,*) "nr_el:", nr_el
	end if

    end subroutine distribute_work

    subroutine set_td_hamiltonian()
	implicit none

	integer(HID_T)			    :: file_id, dset_id, dspace_id, memspace_id
	integer				    :: error
	integer(HSIZE_T), dimension(2)	    :: my_size, max_size, sub_slice
	integer(HSIZE_T), dimension(2)	    :: count
	integer(HSIZE_T), dimension(2)	    :: offset

	!Open stuff.
	call h5open_f(error)
	call h5fopen_f(name_td, H5F_ACC_RDONLY_F, file_id, error)
	call h5dopen_f(file_id, "/couplings", dset_id, error)
	call h5dget_space_f(dset_id, dspace_id, error)
	!---------------------------------------------

	!Get size.
	call h5sget_simple_extent_dims_f(dspace_id, my_size, max_size, error)

	!Set sub_slice size.
	sub_slice(1) = nr_el * nr_vib
	sub_slice(2) = nr_total

	!Allocate my_slice.
	allocate(td_hamiltonian_real(sub_slice(1), sub_slice(2)))
	allocate(td_hamiltonian(sub_slice(1), sub_slice(2)))

	!Create memory dataspace, shaped as the piece to be cut out.
	call h5screate_simple_f(2, sub_slice, memspace_id, error)

	!Hyperslab specs:
	!Starting point of the data rectangle.
	offset(1) = (el_indices(1) - 1) * nr_vib
	offset(2) = 0

	!Lengths of data rectangle.
	count(1)  = nr_el * nr_vib
	count(2)  = nr_total

	call h5sselect_hyperslab_f(dspace_id, H5S_SELECT_SET_F, offset, count, error)

	call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, td_hamiltonian_real, sub_slice, &
	    error, memspace_id, dspace_id)

	!---------------------------------------------
	!Close stuff.
	call h5dclose_f(dset_id, error)
	call h5fclose_f(file_id, error)
	call h5close_f(error)
	

!	!TODO DEBUG!
!	if(my_id == 0) then
!	    write(*,*) "Electronic index: ", el_indices
!	    open(17, file = "td_ham")
!	    write(17,*) td_hamiltonian_real(1,:)
!	    close(17)
!	end if
!	!
	td_hamiltonian = td_hamiltonian_real + complex(0,0)
    end subroutine set_td_hamiltonian
    
    !Initial state.
    !-------------------------------------------------------------------------
    subroutine get_initial_state(name_out, start_index, start_time, el_state, vib_state)
	implicit none
	
	character(len = 150), intent(in)    :: name_out 
	integer, intent(in)		    :: start_index
	real (kind = 8), intent(out)	    :: start_time
	integer, optional, intent(in)	    :: el_state
	integer, optional, intent(in)	    :: vib_state
	integer				    :: el, vib
	
	!HDF5 variables.
	integer(HID_T)			    :: file_id, dset_id 
	integer(HID_T)			    :: dspace_id, memspace_id
	integer				    :: error
	integer(HSIZE_T), dimension(3)	    :: my_size, max_size 
	integer(HSIZE_T), dimension(1)	    :: sub_slice
	integer(HSIZE_T), dimension(3)	    :: count
	integer(HSIZE_T), dimension(3)	    :: offset

	!File reading variables.
	real (kind = 8)			    :: t = 0.0
	real (kind = 8)			    :: field = 0.0
	integer				    :: i
	
	
	!If start_index is one, the initial state is found in the HDF5 file.
	if(start_index == 1) then
	    
	    start_time = 0.0
	    
	    !Assuming ground state if the states are not supplied.
	    if(present(el_state)) then
		el = el_state
	    else
		el = 1
	    end if
	    
	    if(present(vib_state)) then
		vib = vib_state
	    else
		vib = 1
	    end if
	    
	    !Master processor reads HDF5 file for the initial state.
	    if(my_id == 0) then
		!Open stuff.
		call h5open_f(error)
		call h5fopen_f(name_ti, H5F_ACC_RDONLY_F, file_id, error)
		call h5dopen_f(file_id, "/V", dset_id, error)
		call h5dget_space_f(dset_id, dspace_id, error)
		!---------------------------------------------

		!Get size.
		call h5sget_simple_extent_dims_f(dspace_id, my_size, max_size, error)

		!Set sub_slice size.
		sub_slice(1) = nr_vib

		!Create memory dataspace, shaped as the piece to be cut out.
		call h5screate_simple_f(1, sub_slice, memspace_id, error)

		!Hyperslab specs:
		!Starting point of the data rectangle.
		offset(1) = el - 1
		offset(2) = vib - 1
		offset(3) = 0
		
		!Lengths of data rectangle.
		count(1)  = 1
		count(2)  = 1
		count(3)  = nr_vib
		
		!Making sure 
		y = y * 0.0

		call h5sselect_hyperslab_f(dspace_id, H5S_SELECT_SET_F, offset,&
		    count, error)
		
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, &
		    y((el - 1) * nr_vib + 1:el * nr_vib), &
		    sub_slice, error, memspace_id, dspace_id)
		!---------------------------------------------
		!Close stuff.
		call h5dclose_f(dset_id, error)
		call h5fclose_f(file_id, error)
		call h5close_f(error)

		!---------------------------------------------
		!Write initial state to the result file.
		open(unit = 3, file = name_out)
		    write(3,*) t, field, y
		close(3)	

	    end if

	
	else
	    if(my_id == 0) then
		!If this is a restart (i.e. start_index > 1), the old file is
		!rewritten up to start_index, and the latest entry is the new
		!initial state.
		
		!Write to a temporary file.
		open(unit = 5, file = name_out, status = "OLD")
		open(unit = 7, file = "temp")
		
		do i = 1, start_index
		    read(5,*) t, field, y
		    write(7,*) t,field, y
		end do
		
		close(5)
		close(7)
		
		!Write back to the old file name.
		open(unit = 5, file = "temp", status = "OLD")
		open(unit = 7, file = name_out)
		
		do i = 1, start_index
		    read(5,*) t, field, y
		    write(7,*) t,field, y
		end do
		
		close(5)
		close(7)
	    end if
	    
	    start_time = t
	end if

	!Master processor broadcasts it to the whole gang.
	call mpi_bcast(y, 2 * nr_total, MPI_DOUBLE_PRECISION, 0, &
	    MPI_COMM_WORLD, error)
	
	call mpi_bcast(start_time, 1, MPI_DOUBLE_PRECISION, 0, &
	    MPI_COMM_WORLD, error)
	
    end subroutine get_initial_state

    subroutine get_initial_state_from_file(name_out, name_init)
	implicit none
	
	character(len = 120), intent(in)    :: name_out, name_init 
	
	!HDF5 variables.
	integer(HID_T)			    :: file_id, dset_id 
	integer(HID_T)			    :: dspace_id, memspace_id
	integer				    :: error
	integer(HSIZE_T), dimension(1)	    :: my_size, max_size 
	integer(HSIZE_T), dimension(1)	    :: sub_slice
	integer(HSIZE_T), dimension(1)	    :: count
	integer(HSIZE_T), dimension(1)	    :: offset

	!File reading variables.
	real (kind = 8)			    :: t = 0.0
	real (kind = 8)			    :: field = 0.0
	integer				    :: i
	
	    
	!Master processor reads HDF5 file for the initial state.
	if(my_id == 0) then
	    !Open stuff.
	    call h5open_f(error)
	    call h5fopen_f(name_init, H5F_ACC_RDONLY_F, file_id, error)
	    call h5dopen_f(file_id, "/initial_state", dset_id, error)
	    call h5dget_space_f(dset_id, dspace_id, error)
	    !---------------------------------------------

	    !Get size.
	    call h5sget_simple_extent_dims_f(dspace_id, my_size, max_size, error)

	    !Set sub_slice size.
	    sub_slice(1) = 2 * nr_total

	    !Create memory dataspace, shaped as the piece to be cut out.
	    call h5screate_simple_f(1, sub_slice, memspace_id, error)

	    !Hyperslab specs:
	    !Starting point of the data rectangle.
	    offset(1) = 0 
	    
	    !Lengths of data rectangle.
	    count(1)  = 2 * nr_total
	    
	    !Making sure 
	    y = y * 0.0

	    call h5sselect_hyperslab_f(dspace_id, H5S_SELECT_SET_F, offset,&
		count, error)
	    
	    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, &
		y, sub_slice, error, memspace_id, dspace_id)
	    !---------------------------------------------
	    !Close stuff.
	    call h5dclose_f(dset_id, error)
	    call h5fclose_f(file_id, error)
	    call h5close_f(error)

	    !---------------------------------------------
	    !Write initial state to the result file.
	    open(unit = 3, file = name_out)
		write(3,*) t, field, y
	    close(3)	
	
	end if

	!Master processor broadcasts it to the whole gang.
	call mpi_bcast(y, 2 * nr_total, MPI_DOUBLE_PRECISION, 0, &
	    MPI_COMM_WORLD, error)
	
    end subroutine get_initial_state_from_file

    !-------------------------------------------------------------------------
    subroutine set_laser_parameters(input_amplitude, input_omega, &
	input_cycles, input_extra_time)
	!Sets the laser parameters.
	implicit none

	real (kind = 8), intent(in)				:: input_amplitude
	real (kind = 8), intent(in)				:: input_omega
	real (kind = 8), intent(in)				:: input_extra_time
	integer, intent(in)					:: input_cycles

	amplitude = input_amplitude
	omega = input_omega
	cycles = input_cycles
	pulse_duration = 2. * pi /omega * cycles
	total_duration = pulse_duration + input_extra_time
    
    end subroutine set_laser_parameters

    !------------------  Function evaluation - f  ----------------------------
    subroutine f_old(t, y, yp, neqn)
	!Function that will serve as input to ode. 
	!
	!Parameters
	!----------
	!t    : real, the time. Input.
	!y    : real(2 * nr_total), wavefunction coefficients. Input.
	!yp   : real(2 * nr_total), derivative of y. Output.
	!neqn : integer, 2 * nr_total. Input.
	implicit none

	integer, intent(in)				:: neqn
	real (kind = 8), intent(in)			:: t
	real (kind = 8), intent(in), dimension(neqn)	:: y
	real (kind = 8), intent(out), dimension(neqn)	:: yp
	
	complex (kind = 8), dimension(neqn/2)		:: y_complex
	complex (kind = 8), dimension(neqn/2)		:: yp_complex
	complex (kind = 8), dimension(nr_vib)		:: yp_complex_local
	integer						:: i, init, final_local
	integer						:: init_local, final
	integer						:: error_flag
	integer						:: length_message
	real (kind = 8)					:: field
	complex (kind = 8), parameter			:: unit_i =(0.,1.)
	
	!Convert to complex.
	call real_to_complex(y, y_complex, neqn)
	
	!Find the electric field strength.
	call time_function(t, field)
	
	!Time dependent Schroedinger equation.
	!yp = S**-1 * (-i * (H_0 + H(t)) * y)
	do i = 1, nr_el 
	    !Indices.
	    init = (el_indices(i) - 1) * nr_vib + 1 
	    final =  el_indices(i) * nr_vib
	    init_local = (i - 1) * nr_vib + 1 
	    final_local =  i * nr_vib

	    !Time independent part.
	    yp_complex_local = &
		matmul(ti_hamiltonian(i,:,:), y_complex(init:final))
	    
	    !Time dependent part.
	    yp_complex_local = yp_complex_local + &
		matmul(td_hamiltonian(init_local:final_local,:), y_complex) * field
	    
	    !Factors.
	    yp_complex_local = -yp_complex_local * unit_i
	    
	    !Solve for the overlap matrix.
	    call solve_overlap(yp_complex_local, nr_vib)
	    
	end do

	!Length of array to be gathered.
	length_message = 2 * nr_vib

	call mpi_allgather(yp_complex_local, length_message, MPI_COMPLEX, &
	    yp_complex, length_message, MPI_COMPLEX, MPI_COMM_WORLD, error_flag)
	
	!Convert back to real.
	call complex_to_real(yp_complex, yp, nr_total)
    
    end subroutine f_old


    !------------------  Function evaluation - f  ----------------------------
    subroutine f(t, y, yp, neqn)
	!Function that will serve as input to ode. 
	!
	!Parameters
	!----------
	!t    : real, the time. Input.
	!y    : real(2 * nr_total), wavefunction coefficients. Input.
	!yp   : real(2 * nr_total), derivative of y. Output.
	!neqn : integer, 2 * nr_total. Input.
	implicit none

	integer, intent(in)				:: neqn
	real (kind = 8), intent(in)			:: t
	real (kind = 8), intent(in), dimension(neqn)	:: y
	real (kind = 8), intent(out), dimension(neqn)	:: yp
	
	complex (kind = 8), dimension(neqn/2)		:: y_complex
	complex (kind = 8), dimension(neqn/2)		:: yp_complex
	complex (kind = 8), dimension(nr_el * nr_vib)	:: yp_complex_local
	complex (kind = 8), dimension(nr_vib)		:: yp_complex_temp
	integer						:: i, init, final_local
	integer						:: init_local, final
	integer						:: error_flag, j
	integer						:: length_message
	real (kind = 8)					:: field
	complex (kind = 8), parameter			:: unit_i =(0.,1.)

	!Convert to complex.
	call real_to_complex(y, y_complex, neqn)
	
	!Find the electric field strength.
	call time_function(t, field)
	
	!Time dependent Schroedinger equation.
	!yp = S**-1 * (-i * (H_0 + H(t)) * y)
	do i = 1, nr_el  

	    !Indices.
	    init = (el_indices(i) - 1) * nr_vib + 1 
	    final =  el_indices(i) * nr_vib
	    init_local = (i - 1) * nr_vib + 1 
	    final_local =  i * nr_vib
	
	
	    !Time independent part.
	    call matmul_optimizied(ti_hamiltonian(i,:,:),&
	    yp_complex_local(init_local:final_local),&
		y_complex(init:final), order, nr_vib)
	    

	    !Time dependent part.
	    do j = 1, nr_total_el
		call matmul_optimizied(td_hamiltonian( &
		    init_local:final_local,(j-1) * nr_vib + 1:j * nr_vib), &
		    yp_complex_temp, y_complex((j-1) * nr_vib + 1:j * nr_vib)&
		    ,order, nr_vib)
		
		yp_complex_local(init_local:final_local) = &
		    yp_complex_local(init_local:final_local) &
		    + yp_complex_temp * field
	    end do
	    

	    !Factors.
	    yp_complex_local(init_local:final_local) =&
		-yp_complex_local(init_local:final_local) * unit_i

	    !Solve for the overlap matrix.
	    call solve_overlap(yp_complex_local(init_local:final_local), nr_vib)
	    
	end do

	!Length of array to be gathered.
	length_message = 2 * nr_vib

	call mpi_allgather(yp_complex_local, length_message, MPI_COMPLEX, &
	    yp_complex, length_message, MPI_COMPLEX, MPI_COMM_WORLD, error_flag)
	
	
	!Convert back to real.
	call complex_to_real(yp_complex, yp, nr_total)
		
    end subroutine f
    
    !----  Optimalized matrix vector product, for the B-spline submatrix  ----
    subroutine matmul_optimizied(A, b, x, order, len_x)
	implicit none
	
	integer, intent(in)					:: order, len_x
	complex (kind = 8), intent(in), dimension(len_x, len_x)	:: A
	complex (kind = 8), intent(in), dimension(len_x)	:: x
	complex (kind = 8), intent(out), dimension(len_x)	:: b
	
	integer					    :: i, start_index, end_index
	
	do i = 1, len_x
	    start_index = max(i - order + 1, 1)  
	    end_index   = min(i + order - 1, len_x)

	    b(i) = dot_product(	    A(i, start_index:end_index), &
				    x(start_index:end_index))
	end do

    
    end subroutine matmul_optimizied
    
    !-------------------  Solving for the overlap matrix  --------------------
    subroutine solve_overlap(b, size_vector)
	implicit none

	integer, intent(in)			    :: size_vector
	complex (kind = 8), intent(inout), &
    	    dimension(size_vector)		    :: b
	integer					    :: info
	
	call zgetrs("N", size_vector, 1, overlap, size_vector, pivoting, b, &
	    size_vector, info)

    end subroutine solve_overlap

    !------------------------  Conversions  ----------------------------------
    subroutine real_to_complex(real_vector, complex_vector, vector_size)
	!Transforms a real vector into a half as long complex vector.
	implicit none

	integer, intent(in)				    :: vector_size
	real (kind = 8), intent(in), dimension(vector_size) :: real_vector
	complex (kind = 8), intent(out), &
	    dimension(vector_size / 2)			    :: complex_vector
	

	complex_vector =  cmplx(real_vector(1:vector_size / 2), &
				real_vector(vector_size/2 + 1:vector_size))

    end subroutine real_to_complex

    subroutine complex_to_real(complex_vector, real_vector, vector_size)
	!Transforms a complex vector into a twice as long real vector.
	implicit none

	integer, intent(in)				    :: vector_size
	real (kind = 8), intent(out), &
	    dimension(2 * vector_size)			    :: real_vector
	complex (kind = 8), intent(in), &
	    dimension(vector_size)			    :: complex_vector
	
	real_vector(1:vector_size) =  real(complex_vector)
	real_vector(vector_size + 1 : 2 * vector_size) = aimag(complex_vector)

    end subroutine complex_to_real

    !-------------------------  Time function  -------------------------------
    subroutine time_function(t, field)
	!Returns the field strength, <field>, at a given time <t>.
	implicit none
	
	real (kind = 8), intent(in)	:: t
	real (kind = 8), intent(out)	:: field
	real (kind = 8)			:: pi = 3.14159265
	
	if (t <= 1 * pulse_duration) then
	    !Version 1
	    !field = amplitude * sin(pi * t / pulse_duration)**2 * &
	    !	cos(omega * t)
	    
	    !Version 2 - derivative of A.
	    field =  amplitude / omega * (2 * pi / pulse_duration *&
	    		sin(pi * t / pulse_duration) * &
	    	cos(pi * t / pulse_duration) * &
	    	cos(omega * t) -&
	    	sin(pi * t / pulse_duration)**2 * &
	    	sin(omega * t) * &
	    	omega)
	else
	    field = 0.0
	end if
    end subroutine time_function

end module propagator
