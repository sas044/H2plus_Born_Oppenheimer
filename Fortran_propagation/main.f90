program main
    use propagator
    implicit none
    
    !General variables.
    integer					    :: i, file_id = 1


    !ODE variables.
    integer					    :: iflag = 1
    integer, dimension(5)			    :: iwork
    real (kind = 8)				    :: relerr = 1.0e-9
    real (kind = 8)				    :: abserr = 1.0e-9
    real (kind = 8), allocatable, dimension(:)	    :: work 
    real (kind = 8), allocatable, dimension(:)	    :: times

    !Parameters that will be read in from file.
    real (kind = 8)				    :: amp, omg
    real (kind = 8)				    :: ext, start_time
    integer	   				    :: cyc, start_index
    integer	   				    :: el_value, vib_value
    integer	   				    :: timesteps
    character(len = 150)			    :: name_out, name_init



    !Read parameters from file.
    open(unit = 2, file = "input.txt", status="OLD")
    read(2,'(A)') name_out
    read(2,*) amp
    read(2,*) omg
    read(2,*) cyc
    read(2,*) ext
    read(2,*) timesteps
    read(2,*) start_index
    read(2,*) el_value
    read(2,*) vib_value
    read(2,'(A)') name_ti
    read(2,'(A)') name_td
    read(2,'(A)') name_init
    close(2)
    
    call set_laser_parameters( amp, omg, cyc, ext)
    

    call initialize_mpi()
    call set_overlap()
    call set_ti_hamiltonian()
    call set_td_hamiltonian()
    !If the initial state is stored in a the /initial_state group of a separate
    !hdf5 file.
    !call get_initial_state_from_file(name_out, name_init)
    !If the initial state is an eigenstate.
    call get_initial_state(name_out, start_index, start_time, el_value, vib_value)
    
    
    !Setting up the time steps.
    allocate(times(timesteps))
    times = (/ (i , i = 0, timesteps - 1) /)
    !OBS! TODO 2 here AND IN PROPAGATOR TIME FUNCTION!
    times = times * 1 * (total_duration - start_time) / (timesteps - 1.0) + start_time


    !Allocating the work buffer, based on basis sizes given in propagator.mod.
    allocate(work(100 + 21 * 2 * nr_vib * nr_total_el)) 

    if(my_id == 0) then
	open(file_id, file = name_out, status = "OLD", position = "APPEND")
    end if
 
    !Propagating.
    do i = 1, timesteps - 1
	if(my_id == 0 .and. mod(i,10) == 0) then
	    write(*,*) "======", i + 1, " of ", timesteps
	end if

	call propagate(times(i), times(i + 1), y, nr_total * 2)
		
	if(my_id == 0 .and. mod(i,1) == 0) then
	    call write_result(times(i + 1), y, nr_total * 2)
	end if
    end do

    if(my_id == 0) then
	close(file_id)
    end if
	

    call finalize_mpi()

    contains
    !-----------------------  Propagating  -----------------------------------
    subroutine write_result(t, psi, len_psi)
	implicit none
	
	integer					:: len_psi
	real (kind = 8), dimension(len_psi)	:: psi
	real (kind = 8)				:: t, field
	
	call time_function(t, field)

	write(file_id,*) t, field, psi

    end subroutine write_result
    
    
    subroutine propagate(t_0, t_1, y, size_y)
	!Propagates the wavefunction <y> from <t_0> to <t_1>, using ode.
	!
	!Parameters
	!----------
	!t_0    : (in) real, start time.
	!t_1    : (in) real, end_time.
	!y      : (in/out) real array, the wavefunction coefficients. 
	!size_y : (in) integer, the length of y.
	implicit none
	
	integer						    :: counter
	integer, intent(in)				    :: size_y
	real (kind = 8), intent(in)			    :: t_0, t_1
	real (kind = 8), intent(inout), dimension(size_y)   :: y
	
	iflag = 1
	call  ode(f, size_y, y, t_0, t_1, relerr, abserr, iflag, work, iwork)	
	if (iflag .ne. 2) stop
	
    end subroutine propagate
    


end program main
