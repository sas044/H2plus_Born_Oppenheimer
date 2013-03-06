"""
Program for calculating the laser interaction matrix, for the electronic part 
of the H2+ problem, in the Born-Oppenheimer approximation. 
The theoretical foundation can be found in the article
Kamta and Bandrauk, Phys. Rev. A 71, 053407 (2005).

The main method in each class is 'BO_dipole_couplings()'.
When a new gauge/polarisation is needed, add a new subclass of 'TDSE_electron',
as illustrated with 'TDSE_length_z'.
"""

from numpy import zeros, conj, transpose, dot, r_, argmin, array
from pylab import find

import name_generator as name_gen
import config
import psc_basis
import tables
import pypar
import nice_stuff

##############################################################################
#############	          Parent class               #########################
##############################################################################

class TDSE_electron:
    def __init__(self, filename = None, conf = None):
	"""
	Contructor should be further implemented in the subclasses.
	"""
	
	    
	if filename != None:
	    conf = config.Config(filename = filename)

	elif conf != None:
	    pass

	else:
	    raise IOError("Wrong input parameters.")
	
	#Adds the config instance as a class variable.
	self.config = conf

	#Name of HDF5 file where the eigenstates are stored.
	self.eigenstate_file = name_gen.electronic_eigenstates_R(self.config)
	
	#Variables to be filled in the subclasses.
	self.coupling_file = None
	self.laser_info = None
	
    
    def dipole_psc(self, basis):
	"""
	dipole_matrix = dipole_psc(basis)
	
	Calculates the dipole matrix for whichever gauge and orientation is 
	implemented in the subclass.

	Notes
	-----
	The implementation is dependent on the laser orientation and gauge.
	This method is overridden in the subclasses.
	"""
	raise NotImplementedError("Use an appropriate subclass.") 

    def BO_dipole_couplings(self, m, q, E_lim):
	"""
	Parallel method for calculating the Born Oppenheimer dipole couplings
	in the electronic eigenstate basis for a sufficient number of 
	internuclear distances.
	
	Notes
	-----
	The implementation is dependent on the laser orientation and gauge.
	This method is overridden in the subclasses.
	"""
	raise NotImplementedError("Use an appropriate subclass.") 



    def calculate_dipole_eig_R(self, index_array, R):
	"""
	dipole_matrix_eig, E = calculate_dipole_eig_R(index_array, R)

	Calculates the dipole matrix in the eigenstate basis for a given 
	internuclear distance <R>.

	Parameters
	----------
	index_array : (3 x N) int array, where row i contains 
	    [m value, q value, index] of the i-th eigenstate basis function
	    in the eigenstate HDF5 file.
	R : float, the internuclear distance for which the dipole matrix 
	    should be calculated.
	
	Returns
	-------
	dipole_matrix_eig : (N x N) complex array, containing the dipole 
	    matrix in the eigenstate basis.
	E : 1D float array, containing the energies of the eigenstate basis 
	    function.
	"""

	#Create the correct basis.
	temp_config = config.Config(self.config)
	temp_config.R = R
	basis = psc_basis.PSC_basis(temp_config)

	#Computes the dipole matrix in the psc basis.
	dipole_matrix_psc = self.dipole_psc(basis)
	
	#Initialize eigenstate/value arrays.
	V = zeros([basis.basis_size, index_array.shape[0]], dtype = complex)
	E = zeros([index_array.shape[0]])

	f = tables.openFile(self.eigenstate_file)
	try:
	    #Find index of current R in the R_grid.
	    R_index = find(f.root.R_grid[:] == R)[0]

	    for i, [m, q, index] in enumerate(index_array):
		m_group = name_gen.m_name(m)
		q_group = name_gen.q_name(q)
		V[:,i] = eval("f.root.%s.%s.V[:, %i, %i]"%(
		    m_group, q_group, index, R_index))
		E[i] = eval("f.root.%s.%s.E[%i, %i]"%(
		    m_group, q_group, index, R_index))
	finally:
	    f.close()
	
	#Transform the dipole matrix to the eigenstate basis.
	dipole_matrix_eig = dot(conj(transpose(V)), dot(dipole_matrix_psc, V))
	
	return  dipole_matrix_eig, E

    def save_dipole_eig_R(self, dipole_matrix_eig, E, R):
	"""
	save_dipole_eig_R(dipole_matrix_eig, E, R)

	Saves <dipole_matrix_eig> and <E> into the HDF5 file for couplings, in the 
	slot appropriate to <R>.

	Parameters
	----------
	dipole_matrix_eig : (N x N) complex array, containing the dipole 
	    matrix in the eigenstate basis.
        E : 1D float array, containing the energies of the eigenstate basis 
	    function.
	R : float, the internuclear distance for which the dipole matrix 
	    was calculated.
	"""
	f = tables.openFile(self.coupling_file, 'r+')
	try:
	    #Find index of current R in the R_grid.
	    R_index = find(f.root.R_grid[:] == R)[0]
	    
	    #Save the couplings and the energies.
	    f.root.couplings[:,:,R_index] = dipole_matrix_eig
	    f.root.E[:,R_index] = E

	finally:
	    f.close()




##############################################################################
#############	    Length gauge, z direction        #########################
##############################################################################

class TDSE_length_z(TDSE_electron):
    
    def __init__(self, filename = None, conf = None):
	"""
	TDSE_length_z(filename = None, conf = None):

	Constructor of an instance of the TDSE_length_z class.

	Parameters
	----------
	    
	    Option I
	    --------
	    filename : string, name of the eigenstate HDF5 file.
	    
	    Option II
	    --------
	    conf : an instance of the Config class.
	"""
	#Utilizing parent class constructor.
	TDSE_electron.__init__(self, filename, conf)

	#Adding laser 'name tag'.
	self.laser_info = "lenght_z" 
	

    
    def dipole_psc(self, basis):
	"""
	dipole_matrix = dipole_length_z(basis)	

	For a given PSC_basis, this method calculates the dipole matrix, for 
	length gauge, i.e. the hamiltonian matrix for the operator H = z.
	The matrix elements are described in equation (B2) in Kamta2005.

	Parameters
	----------
	basis : a PSC_basis instance. The basis the dipole matrix should be 
	    represented in.

	Returns
	-------
	dipole_matrix : 2D complex array, the dipole matrix in the PSC basis.
	"""

	#Initialize overlap matrix.
	dipole_matrix = zeros([basis.basis_size, basis.basis_size], 
	    dtype = complex)
	
	#Looping over indices.
	#<bra|
	for i, [m_prime,nu_prime,mu_prime] in enumerate(basis.index_iterator):
	    #|ket>
	    for j, [m, nu, mu] in enumerate(basis.index_iterator):
		#Selection rule.
		if m_prime == m:
		    #Upper triangular part of the matrix.
		    if j >= i:
			dipole_matrix[i,j] = (basis.config.R/2.)**4 * (
			      basis.find_d(3, m, nu_prime, nu) 
			    * basis.find_d_tilde(1, m, mu_prime, mu)
			    - basis.find_d(1, m, nu_prime, nu) 
			    * basis.find_d_tilde(3, m, mu_prime, mu))
		    else:
			#Lower triangular part is equal to the upper part.
			    #TODO Should this be the conjugated? Yes?? No??
			    #Might not be tht simple. Possibly (A6) & (A8) in
			    #Kamta2005 should be modified.
			dipole_matrix[i,j] = conj(dipole_matrix[j,i])
	
	return dipole_matrix



    def BO_dipole_couplings(self, m_list, q_list, E_lim):
	"""
	BO_dipole_couplings(m_list, q_list, E_lim)

	Parallel program that calculates the dipole couplings for a 
	z-polarized laser in lenght gauge. An eigenstate basis is used, of 
	states whose quantum numbers are in <m_list> and <q_list>, that have 
	energies below <E_lim>. The couplings are stored to an HDF5 file.

	Parameters
	----------
	m_list : list of integers, containing the m values wanted in 
	    the basis.
	q_list : list of integers, containing the q values wanted in 
	    the basis.
	E_lim : float, the upper limit of the energies wanted in 
	    the basis, for R ~ 2.0.

	Notes
	-----
	I sometimes observe unnatural spikes in the couplings 
	(as a function of R), which should be removed before the couplings 
	are used. I don't know why they are there.    

	Example
	-------
	>>> filename = "el_states_m_0_nu_70_mu_25_beta_1_00_theta_0_00.h5"
	>>> tdse = tdse_electron.TDSE_length_z(filename = filename)
	>>> m = [0]
	>>> q = [0,1,2,3]
	>>> E_lim = 5.0
	>>> tdse.BO_dipole_couplings(m, q, E_lim)
	"""
	#Name of the HDF5 file where the couplings will be saved.
	self.coupling_file = name_gen.electronic_eig_couplings_R(self, 
	    m_list, q_list, E_lim)

	#Parallel stuff
	#--------------
	#Get processor 'name'.
	my_id = pypar.rank() 
	
	#Get total number of processors.
	nr_procs = pypar.size()

	#Size of eigenstate basis. (Buffer for broadcast.)
	basis_size_buffer = r_[0]

	#Get number of tasks.
	f = tables.openFile(self.eigenstate_file)
	try:
	    R_grid = f.root.R_grid[:]
	finally:
	    f.close()
	
	nr_tasks = len(R_grid)

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
	    
	    #Initialize index list.
	    index_array = []

	    #Find the index of the R closest to 2.0.
	    R_index = argmin(abs(R_grid - 2.0))
	    
	    #Choose basis functions.
	    f = tables.openFile(self.eigenstate_file)
	    try:
		for m in m_list:
		    m_group = name_gen.m_name(m)
		    for q in q_list:
			q_group = name_gen.q_name(q)
			for i in range(self.config.nu_max + 1):
			    if eval("f.root.%s.%s.E[%i,%i]"%(m_group, q_group, 
				i, R_index)) > E_lim:
				break
			    else:
				#Collect indices of the basis functions.
				index_array.append(r_[m, q, i])
	    finally:
		f.close()
	    
	    #Cast index list as an array.
	    index_array = array(index_array)
	    
	    #Number of eigenstates in the basis.
	    basis_size = len(index_array)
	    print basis_size, "is the basis size"
	    basis_size_buffer[0] = basis_size

	    f = tables.openFile(self.coupling_file, 'w')
	    try:
		f.createArray("/", "R_grid", R_grid)
		
		#Saving the index array.
		f.createArray("/", "index_array", index_array)
		
		#Initializing the arrays for the couplings and energies.
		f.createCArray('/', 'E', 
		    tables.atom.FloatAtom(), 
		    (basis_size, nr_tasks),
		    chunkshape=(basis_size, 1))
		
		f.createCArray('/', 'couplings', 
		    tables.atom.ComplexAtom(16), 
		    (basis_size, basis_size, nr_tasks),
		    chunkshape=(basis_size, basis_size, 1))
		
	    finally:
		f.close()
	    
	    #Save config instance.
	    self.config.save_config(self.coupling_file)
	#----------------------------------


	#Calculating the dipole couplings
	#--------------------------------
	#Broadcasting the basis size from processor 0.
	pypar.broadcast(basis_size_buffer, 0)

	#Initializing the index array.
	if my_id != 0:
	    index_array = zeros([basis_size_buffer[0], 3], dtype=int)
	
	#Broadcasting the index array from proc. 0.
	pypar.broadcast(index_array, 0)


	#Looping over the tasks of this processor.
	for i in my_tasks:

	    #Calculate the dipole couplings for one value of R.
	    couplings, E = self.calculate_dipole_eig_R(index_array, R_grid[i])


	    #First file write. (Send, but not receive baton.)
	    if starter == my_id:
		#Write to file.
		self.save_dipole_eig_R(couplings, E, R_grid[i])
		
		#Avoiding this statement 2nd time around.
		starter = -1

		#Sending the baton to the next writer.
		pypar.send(baton, send_to, use_buffer = True)

	    
	    #Last file write. (Receive, but not send baton.)
	    elif i == my_tasks[-1] and ender == my_id :
		#Receiving the baton from the previous writer.
		pypar.receive(receive_from, buffer = baton)

		#Write to file.
		self.save_dipole_eig_R(couplings, E, R_grid[i])
	    
	    #The rest of the file writes.
	    else:
		#Receiving the baton from the previous writer.
		pypar.receive(receive_from, buffer = baton)

		#Write to file.
		self.save_dipole_eig_R(couplings, E, R_grid[i])

		#Sending the baton to the next writer.
		pypar.send(baton, send_to, use_buffer = True)
	    
	    
	    #Showing the progress of the work.
	    if my_id == 0:
		nice_stuff.status_bar("Electronic dipole couplings:", 
		    i, len(my_tasks))
	#----------------------------
	
	#Letting everyone catch up. 
	pypar.barrier()




	    


