"""
Solving the vibrational part of the H2+ problem in the Born Oppenheimer
approximation. 

One can either save the vibrational eigenstates, 
or save the full laser interaction hamiltonian.

Running the main() method will execute both of these programs.
The problem parameters should be filled in in the problem_parameters() method.
"""

from numpy import r_, diff, copy, average
from pylab import find

import vibrational_methods
import name_generator as name_gen
import nice_stuff
import tables
import pypar

def problem_parameters():
    """
    filename_el, nr_kept, xmin, xmax, xsize, order = problem_parameters()

    Write your preferred parameters in this method.
    """
    #Name of the file where the electronic couplings are stored.
    filename_el = ("el_couplings_lenght_z_m_0_nu_25_mu_15_beta_1_00_" + 
	"theta_0_00_m_0_q_3_E_5_00.h5")
    
    #Number of vibrational states to be kept when diagonalizing.
    nr_kept = 50

    #Start of the R grid.
    xmin = 0.5
    
    #End of the R grid.
    xmax = 20

    #Approximat number of B-splines.
    xsize = 100

    #Order of the B-splines.
    order = 6

    return  filename_el, nr_kept, xmin, xmax, xsize, order

#TODO REMOVE? Causes errors. Removes this, and prays for no spikes.
#def remove_point(vector_1, i):
#    """
#    vector_2 = remove_point(vector_1, i)
#
#    Replace element <i> in <vector_1> with the average of the neighbouring 
#    points.
#
#    Parameters
#    ----------
#    vector_1 : 1D float array.
#
#    Returns
#    -------
#    vector_2 : 1D float array.     
#    """
#    if i == 0:
#        new_point = vector_1[i+1]
#    elif i == len(vector_1) - 1:
#        new_point = vector_1[i-1] 
#    else:
#        new_point = (vector_1[i-1] + vector_1[i+1])/2.
#	
#    vector_2 = copy(vector_1)
#    vector_2[i] = new_point
#    
#    return vector_2
#
#TODO REMOVE?
def remove_spikes_X(r_grid, coupling):
    """
    smooth_coupling = remove_spikes(coupling)

    Removes the spikes that unfortunately mar some of the electronic 
    couplings. Not sure why they appear, or if they can be rooted out 
    earlier. (Edit: Most disappeared with a fix of the parallelization.) 

    Parameters
    ----------
    coupling : 1D float array, (perhaps with spikes).

    Returns
    -------
    smooth_coupling : 1D float array, (no spikes). 
    """

    smooth_coupling = copy(coupling)
    r_copy = copy(r_grid)

    #The spike is easiest found in the derivated.
    diff_coupling = abs(diff(smooth_coupling))

    #Somewhat arbitrary limit for what IS a spike.
    limit = 2 * average(diff_coupling)
    indices = find(diff_coupling > limit)
    
    #A spike typically means two large entries in the derivative.
    indices_2 = find(diff(indices) == 1)

    remove_these = indices[indices_2 + 1]
    
    all_indices = range(len(smooth_coupling))
    keep_these = list(set(all_indices).difference(set(remove_these)))

    return r_copy[keep_these], smooth_coupling[keep_these]
#    #Replace with average of neighbouring points.
#    for i in remove_these:
#	smooth_coupling = remove_point(smooth_coupling,i)
#    
#    return smooth_coupling

def remove_spikes(r_grid, coupling):
    """
    smooth_coupling = remove_spikes(coupling)

    Removes the spikes that unfortunately mar some of the electronic 
    couplings. Not sure why they appear, or if they can be rooted out 
    earlier. (Edit: Most disappeared with a fix of the parallelization.) 

    Parameters
    ----------
    coupling : 1D float array, (perhaps with spikes).

    Returns
    -------
    smooth_coupling : 1D float array, (no spikes). 
    """

    smooth_coupling = copy(coupling)
    r_copy = copy(r_grid)

    #The spike is easiest found in the derivated.
    diff_coupling = diff(smooth_coupling) 
    diff_coupling *= abs(diff(smooth_coupling)) > (3 * average(abs(diff_coupling))) 
    diff_coupling = diff(diff_coupling) > 0
    

    indices_1 = find((diff_coupling[:-2]  == 0) & 
		     (diff_coupling[1:-1] == 1) & 
		     (diff_coupling[2:]   == 0)) + 2
    
    indices_2 = find((diff_coupling[:-2]  == 1) & 
		     (diff_coupling[1:-1] == 0) & 
		     (diff_coupling[2:]   == 1)) + 2
    
    remove_these = r_[indices_1 , indices_2]
    
    all_indices = range(len(smooth_coupling))
    keep_these = list(set(all_indices).difference(set(remove_these)))

    return r_copy[keep_these], smooth_coupling[keep_these]



def save_all_eigenstates(filename_el, nr_kept, xmin, xmax, xsize, order):
    """
    save_all_eigenstates(filename_el, nr_kept, xmin, xmax, xsize, order)

    This program solves the vibrational TISE for a set of energy curves, 
    and stores them in an HDF5 file. 
    This program must be run in parallel.
    
    Example
    -------
    To run this program on 5 processors:
    
    $ mpirun -n 5 python -c "execfile('vibrational_BO.py');save_eigenstates()"
    """

    #Retrieve the electronic energy curves.
    f = tables.openFile(filename_el)
    try:
	r_grid = f.root.R_grid[:]
	energy_curves = f.root.E[:]

    finally:
	f.close()
    
    #Initialize the B-spline_basis.
    spline_basis = vibrational_methods.Bspline_basis(xmin, xmax, xsize, order)
    spline_basis.setup_kinetic_hamiltonian()
    spline_basis.setup_overlap_matrix()
    
    #Generate a filename.
    filename = name_gen.vibrational_eigenstates(filename_el, spline_basis)
    
    #Parallel stuff
    #--------------
    #Get processor 'name'.
    my_id = pypar.rank() 
    
    #Get total number of processors.
    nr_procs = pypar.size()
    
    #Get number of tasks.
    nr_tasks = len(energy_curves)

    #Get a list of the indices of this processors share of R_grid. 
    my_tasks = nice_stuff.distribute_work(nr_procs, nr_tasks, my_id)

    #The processors will be writing to the same file.
    #In order to avoid problems, the procs will do a relay race of writing to
    #file. This is handeled by blocking send() and receive().
    #Hopefully there will not be to much waiting.

    #ID of the processor that will start writing.
    starter = 0

    #ID of the processor that will be the last to write.
    ender = (nr_tasks - 1) % nr_procs

    #Buffer for the baton, i.e. the permission slip for file writing.
    baton = r_[0]

    #The processor one is to receive the baton from.
    receive_from = (my_id - 1) % nr_procs 

    #The processor one is to send the baton to.
    send_to = (my_id + 1) % nr_procs 
    #-------------------------------

    
    #Initializing the HDF5 file
    #--------------------------
    if my_id == 0:

	f = tables.openFile(filename, 'w')
	try:
	    f.createArray("/", "electronicFilename", [filename_el])	    
	    
	    f.createArray("/", "R_grid", r_grid)	    
	    
	    f.createArray("/", "overlap", spline_basis.overlap_matrix)	    
	    
	    #Initializing the arrays for the eigenvalues and states.
	    f.createCArray('/','E', 
		tables.atom.FloatAtom(), 
		(nr_kept, nr_tasks),
		chunkshape=(nr_kept, 1))
	    
	    f.createCArray('/','V', 
		tables.atom.FloatAtom(), 
		(spline_basis.nr_splines, nr_kept, nr_tasks),
		chunkshape=(spline_basis.nr_splines, nr_kept, 1))
	    
	    f.createCArray('/','hamiltonian', 
		tables.atom.FloatAtom(), 
		(spline_basis.nr_splines, spline_basis.nr_splines, nr_tasks),
		chunkshape=(spline_basis.nr_splines, spline_basis.nr_splines, 
		1))
	    

	    
	finally:
	    f.close()
	
	#Save spline info.
	spline_basis.bsplines.save_spline_info(filename)
    #----------------------------------

    #Solving the TISE
    #----------------
    #Looping over the tasks of this processor.
    for i in my_tasks:

	
	#TODO REMOVE?
	#remove_spikes removes points where the diagonalization has failed.
	#potential_hamiltonian = spline_basis.setup_potential_matrix(
	#    r_grid, remove_spikes(energy_curves[i,:]) + 1/r_grid)
	####

	#Setup potential matrix. 
	potential_hamiltonian = spline_basis.setup_potential_matrix(
	    r_grid, energy_curves[i,:] + 1/r_grid)
		
	#The total hamiltonian.
	hamiltonian_matrix = (spline_basis.kinetic_hamiltonian + 
	    potential_hamiltonian)

	#Diagonalizing the hamiltonian.
	E, V = spline_basis.solve(hamiltonian_matrix, nr_kept)
	
	#First file write. (Send, but not receive baton.)
	if starter == my_id:
	    #Write to file.
	    spline_basis.save_eigenstates(filename, E, V, 
		hamiltonian_matrix, i)

	    #Avoiding this statement 2nd time around.
	    starter = -1

	    #Sending the baton to the next writer.
	    pypar.send(baton, send_to, use_buffer = True)
	
	#Last file write. (Receive, but not send baton.)
	elif i == my_tasks[-1] and ender == my_id :
	    #Receiving the baton from the previous writer.
	    pypar.receive(receive_from, buffer = baton)

	    #Write to file.
	    spline_basis.save_eigenstates(filename, E, V, 
		hamiltonian_matrix, i)
	
	#The rest of the file writes.
	else:
	    #Receiving the baton from the previous writer.
	    pypar.receive(receive_from, buffer = baton)

	    #Write to file.
	    spline_basis.save_eigenstates(filename, E, V,
		hamiltonian_matrix, i)

	    #Sending the baton to the next writer.
	    pypar.send(baton, send_to, use_buffer = True)
	
	
	#Showing the progress of the work.
	if my_id == 0:
	    nice_stuff.status_bar("Vibrational BO calculations", 
		i, len(my_tasks))
    #----------------------------

    #Letting everyone catch up. 
    pypar.barrier()


def save_all_couplings(filename_el, nr_kept, xmin, xmax, xsize, order):
    """
    save_all_couplings(filename_el, nr_kept, xmin, xmax, xsize, order)

    This program sets up the laser interaction hamiltonian, 
    and stores it in an HDF5 file. 
    This program must be run in parallel.
    
    Example
    -------
    To run this program on 5 processors:
    
    $ mpirun -n 5 python -c "execfile('vibrational_BO.py');save_eigenstates()"
    """
    
    #Retrieve the electronic energy curves.
    f = tables.openFile(filename_el)
    try:
	r_grid = f.root.R_grid[:]
	#Get number of tasks.
	el_basis_size = f.root.couplings.shape[0]
    finally:
	f.close()
    
    #Filter function, describing what index pairs should be included in the 
    #calculations.
    def no_filter(index_pair):
	"""
	All couplings included.
	"""
	return True
    
    def symmetry_filter(index_pair):
	"""
	Only include the upper/lower triangular, since the hermeticity means
	that they are the same.
	"""
	i = index_pair[0]
	j = index_pair[1]
	if i >= j:
	    return True
	else:
	    return False    
    
    #Make a list of the coupling indices that should be included.
    index_table = create_index_table(el_basis_size, no_filter)
    nr_tasks = len(index_table)

    #Initialize the B-spline_basis.
    spline_basis = vibrational_methods.Bspline_basis(xmin, xmax, xsize, order)
    vib_basis_size = spline_basis.nr_splines

    #Generate a filename.
    filename = name_gen.couplings(filename_el, spline_basis)
    
    #Parallel stuff
    #--------------
    #Get processor 'name'.
    my_id = pypar.rank() 
    
    #Get total number of processors.
    nr_procs = pypar.size()
    
    #Get a list of the indices of this processors share of R_grid. 
    my_tasks = nice_stuff.distribute_work(nr_procs, nr_tasks, my_id)

    #The processors will be writing to the same file.
    #In order to avoid problems, the procs will do a relay race of writing to
    #file. This is handeled by blocking send() and receive().
    #Hopefully there will not be to much waiting.

    #ID of the processor that will start writing.
    starter = 0

    #ID of the processor that will be the last to write.
    ender = (nr_tasks - 1) % nr_procs

    #Buffer for the baton, i.e. the permission slip for file writing.
    baton = r_[0]

    #The processor one is to receive the baton from.
    receive_from = (my_id - 1) % nr_procs 

    #The processor one is to send the baton to.
    send_to = (my_id + 1) % nr_procs 
    #-------------------------------

    
    #Initializing the HDF5 file
    #--------------------------
    if my_id == 0:

	f = tables.openFile(filename, 'w')
	try:
	    f.createArray("/", "electronicFilename", [filename_el])	    
	    
	    #Initializing the arrays for the eigenvalues and states.
	    f.createCArray('/','couplings', 
		tables.atom.FloatAtom(), 
		(vib_basis_size * el_basis_size, 
		 vib_basis_size * el_basis_size),
		chunkshape=(vib_basis_size, vib_basis_size))
	    
	finally:
	    f.close()
	
	#Save spline info.
	spline_basis.bsplines.save_spline_info(filename)
    #----------------------------------


    #Setting up the hamiltonian
    #--------------------------
    #Looping over the tasks of this processor.
    for i in my_tasks:
	#Retrieve indices.
	row_index, column_index = index_table[i]

	#Retrieve electronic couplings.
	f = tables.openFile(filename_el)
	try:
	    el_coupling = f.root.couplings[row_index, column_index,:]
	finally:
	    f.close()
	
	#TODO REMOVE?
	#Remove errors from the coupling. (A hack, unfortunately.) 
	r_grid_2, el_coupling_2 = remove_spikes(r_grid, el_coupling)

	#Setup potential matrix.
	couplings = spline_basis.setup_potential_matrix(
	    r_grid_2, el_coupling_2)
	
	#First file write. (Send, but not receive baton.)
	if starter == my_id:
	    #Write to file.
	    spline_basis.save_couplings(filename, couplings, 
		row_index, column_index)

	    #Avoiding this statement 2nd time around.
	    starter = -1

	    #Sending the baton to the next writer.
	    pypar.send(baton, send_to, use_buffer = True)
	
	#Last file write. (Receive, but not send baton.)
	elif i == my_tasks[-1] and ender == my_id :
	    #Receiving the baton from the previous writer.
	    pypar.receive(receive_from, buffer = baton)

	    #Write to file.
	    spline_basis.save_couplings(filename, couplings, 
		row_index, column_index)
	
	#The rest of the file writes.
	else:
	    #Receiving the baton from the previous writer.
	    pypar.receive(receive_from, buffer = baton)

	    #Write to file.
	    spline_basis.save_couplings(filename, couplings, 
		row_index, column_index)

	    #Sending the baton to the next writer.
	    pypar.send(baton, send_to, use_buffer = True)
	
	
	#Showing the progress of the work.
	if my_id == 0:
	    nice_stuff.status_bar("Calculating couplings:", 
		i, len(my_tasks))
    #----------------------------

    #Letting everyone catch up. 
    pypar.barrier()


def save_eigenfunction_couplings(filename_el, nr_kept, xmin, xmax, xsize, order):
    """
    save_eigenfunction_couplings(filename_el, nr_kept, xmin, xmax, xsize, order)

    This program sets up the laser interaction hamiltonian for the 
    eigenfunction basis, and stores it in an HDF5 file. 
    This program must be run in parallel.
    
    Example
    -------
    To run this program on 5 processors:
    
    $ mpirun -n 5 python -c "execfile('vibrational_BO.py');save_eigenstates()"
    """
    
    #Retrieve the electronic energy curves.
    f = tables.openFile(filename_el)
    try:
	r_grid = f.root.R_grid[:]
	#Get number of tasks.
	el_basis_size = f.root.couplings.shape[0]
    finally:
	f.close()
    
    #Filter function, describing what index pairs should be included in the 
    #calculations.
    def no_filter(index_pair):
	"""
	All couplings included.
	"""
	return True
    
    def symmetry_filter(index_pair):
	"""
	Only include the upper/lower triangular, since the hermeticity means
	that they are the same.
	"""
	i = index_pair[0]
	j = index_pair[1]
	if i >= j:
	    return True
	else:
	    return False    
    
    #Make a list of the coupling indices that should be included.
    index_table = create_index_table(el_basis_size, no_filter)
    nr_tasks = len(index_table)

    #Initialize the B-spline_basis.
    spline_basis = vibrational_methods.Bspline_basis(xmin, xmax, xsize, order)
    vib_basis_size = spline_basis.nr_splines
    
    #Generate a filename.
    filename = name_gen.eigenfunction_couplings(filename_el, spline_basis)

    #Name of vib states.
    filename_vib = name_gen.vibrational_eigenstates(filename_el, spline_basis)
    
    #Parallel stuff
    #--------------
    #Get processor 'name'.
    my_id = pypar.rank() 
    
    #Get total number of processors.
    nr_procs = pypar.size()
    
    #Get a list of the indices of this processors share of R_grid. 
    my_tasks = nice_stuff.distribute_work(nr_procs, nr_tasks, my_id)

    #The processors will be writing to the same file.
    #In order to avoid problems, the procs will do a relay race of writing to
    #file. This is handeled by blocking send() and receive().
    #Hopefully there will not be to much waiting.

    #ID of the processor that will start writing.
    starter = 0

    #ID of the processor that will be the last to write.
    ender = (nr_tasks - 1) % nr_procs

    #Buffer for the baton, i.e. the permission slip for file writing.
    baton = r_[0]

    #The processor one is to receive the baton from.
    receive_from = (my_id - 1) % nr_procs 

    #The processor one is to send the baton to.
    send_to = (my_id + 1) % nr_procs 
    #-------------------------------

    
    #Initializing the HDF5 file
    #--------------------------
    if my_id == 0:

	f = tables.openFile(filename, 'w')
	g = tables.openFile(filename_vib)
	try:
	    f.createArray("/", "electronicFilename", [filename_el])	    
	    
	    #Initializing the arrays for the time dependent couplings of H.
	    f.createCArray('/','couplings', 
		tables.atom.FloatAtom(), 
		(nr_kept * el_basis_size, 
		 nr_kept * el_basis_size),
		chunkshape=(nr_kept,nr_kept))
	    
	    #Energy diagonal. Time independent part of H. 
	    energy_diagonal = zeros(nr_kept * el_basis_size)
	    for i in range(el_basis_size):
		energy_diagonal[nr_kept * i:
		    nr_kept * (i + 1)] = g.root.E[:nr_kept,i]

	    f.createArray("/", "energyDiagonal", energy_diagonal)
	    
	finally:
	    f.close()
	    g.close()
	
	#Save spline info.
	spline_basis.bsplines.save_spline_info(filename)
    #----------------------------------


    #Setting up the hamiltonian
    #--------------------------
    #Looping over the tasks of this processor.
    for i in my_tasks:
	#Retrieve indices.
	row_index, column_index = index_table[i]

	#Retrieve electronic couplings.
	f = tables.openFile(filename_el)
	try:
	    el_coupling = f.root.couplings[row_index, column_index,:]
	finally:
	    f.close()
	
#	#TODO REMOVE?
#	#Remove errors from the coupling. (A hack, unfortunately.) 
#	r_grid_2, el_coupling_2 = remove_spikes(r_grid, el_coupling)
#
#	#Setup potential matrix.
#	couplings = spline_basis.setup_potential_matrix(
#	    r_grid_2, el_coupling_2)
#	
	#Setup potential matrix. Aij = <Bi | f(R) | Bj>
	bfb_matrix = spline_basis.setup_potential_matrix(
	    r_grid, el_coupling)
	
	couplings = zeros([nr_kept, nr_kept])
	
	#Retrieve eigensvectors.
	g = tables.openFile(filename_vib)
	try:
	    Vr = g.root.V[:,:,row_index] 
	    Vc = g.root.V[:,:,column_index]
	finally:
	    g.close()
	
	#Calculate couplings.
	for r_index in range(nr_kept):
	    for c_index in range(nr_kept):
		couplings[r_index, c_index] = dot(Vr[:,r_index], 
		    dot(Vc[:,c_index]))
	
	

	#First file write. (Send, but not receive baton.)
	if starter == my_id:
	    #Write to file.
	    spline_basis.save_couplings(filename, couplings, 
		row_index, column_index)

	    #Avoiding this statement 2nd time around.
	    starter = -1

	    #Sending the baton to the next writer.
	    pypar.send(baton, send_to, use_buffer = True)
	
	#Last file write. (Receive, but not send baton.)
	elif i == my_tasks[-1] and ender == my_id :
	    #Receiving the baton from the previous writer.
	    pypar.receive(receive_from, buffer = baton)

	    #Write to file.
	    spline_basis.save_couplings(filename, couplings, 
		row_index, column_index)
	
	#The rest of the file writes.
	else:
	    #Receiving the baton from the previous writer.
	    pypar.receive(receive_from, buffer = baton)

	    #Write to file.
	    spline_basis.save_couplings(filename, couplings, 
		row_index, column_index)

	    #Sending the baton to the next writer.
	    pypar.send(baton, send_to, use_buffer = True)
	
	
	#Showing the progress of the work.
	if my_id == 0:
	    nice_stuff.status_bar("Calculating couplings:", 
		i, len(my_tasks))
    #----------------------------

    #Letting everyone catch up. 
    pypar.barrier()


def create_index_table(basis_size, filter_function):
    """
    index_table = create_index_table(basis_size, filter_function)

    Creates a table of the index combinations that are wanted.

    Parameters
    ----------
    basis_size : integer, the length of the electronic coupling matrix.
    filter_function : function, returns a boolean based on the input i & j.

    Returns
    -------
    index_table : list, containing valid index pairs.
    """
    index_table = []
    
    #Create a full index table.
    for i in range(basis_size):
	for j in range(basis_size):
	    index_table.append(r_[i,j])
    
    #Weed out the unwanted.
    filtered_table = filter(filter_function, index_table)
    
    return filtered_table




