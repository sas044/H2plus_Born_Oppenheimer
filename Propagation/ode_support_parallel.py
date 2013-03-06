from scipy.linalg import lu_factor, lu_solve
from numpy import zeros, pi, sin, cos, dot, r_, real, imag, transpose, save
from pylab import shape

import sys
sys.path.append("..")
import tables
import pypar
import time
import nice_stuff

class Ode_function_call:
    """
    This class provides a function that can serve as input to 
    scipy.integrate.odeint().
    """
    
    def __init__(self, filename_coupling, filename_vib, 
	amplitude, omega, cycles, extra_time):
	"""
	Ode_function_call(filename_coupling, filename_vib,
	    amplitude, omega, cycles,extra_time)

	Constructor.

	Parameters
	----------
	filename_coupling : string, path to the HDF5 file that contains the 
	    dipole couplings of the H2+ BO problem.
	filename_vib : string, path to the HDF5 file that contains the 
	    vibrational couplings of the H2+ BO problem.
	"""
	#Laser info.
	self.amplitude = amplitude
	self.omega = omega
	self.cycles = cycles
	self.pulse_duration = 2 * pi /(self.omega) * self.cycles
	self.total_duration = self.pulse_duration + extra_time

	
	#Retrieve coupling information.	
	f = tables.openFile(filename_vib)
	g = tables.openFile(filename_coupling)
    	
	#Find the cholesky factorization of the overlap matrix.
	self.overlap_fact = lu_factor(f.root.overlap[:])


	#Basis sizes.
	self.vib_basis_size = f.root.hamiltonian.shape[0]
	self.basis_size = g.root.couplings.shape[0]
	self.el_basis_size = self.basis_size / self.vib_basis_size

	
	#Parallel stuff.
	self.my_id = pypar.rank()
	self.nr_procs = pypar.size()
	
	#Assert that the electronic basis size is dividable by nr_procs.
	assert self.el_basis_size%self.nr_procs == 0

	self.my_tasks = nice_stuff.distribute_work(
	    self.nr_procs, self.el_basis_size, self.my_id)
	

	
	#Time indepandent Hamiltonian.
	self.h_0 = f.root.hamiltonian[:,:,self.my_tasks]

	#Time dependent hamiltonian.
	self.my_slice = slice(self.my_tasks[0] * self.vib_basis_size, 
	    (self.my_tasks[-1] + 1) * self.vib_basis_size)

	self.h_1 = self.retrieve_hamiltonian(g)

	f.close()
	g.close()
	
	#Time keeping.
	self.t_0 = time.time()

#	#Avoid repeating steps.
#	######################
#	self.prev_t = 0.
#	self.prev_in = 0.
#	self.prev_out = 0.
#	self.repeats = 0
#	######################

#    def check_novelty(self, t, psi):
#	"""
#
#	"""
#	if abs(sum((psi)) - sum((self.prev_in))) < 1e-13 and t == self.prev_t: 
#	    novel = False
#	    self.repeats += 1
#	    if self.repeats > 10:
#		save("aborted_t", self.prev_t)
#		save("aborted_psi", self.prev_in)
#		raise
#	    return novel, self.prev_out
#	else:
#	    novel = True
#	    self.prev_t = t
#	    self.prev_in = psi
#	    self.repeats = 0
#	    return novel, None

    
    def debug_norm(self, t, v_0, v_1):
	f = open("norm_testing_1.txt",'a')
	f.write("[%2.10f,%2.10f,%2.10f],\n"%(t,sum(v_0),sum(v_1)))
	f.close()

    def dpsi_dt(self, psi, t):
	"""
	dp_dt = dpsi_dt(psi, t)

	Method that serves as input to the odeint() function.
	Calculates dpsi/dt = -i S**-1 H(t) psi.

	Parameters
	----------
	psi : 1D complex array. Wavefunction.
	t : float. Time.

	Returns
	-------
	dp_dt : 1D complex array. Derivative of the wavefunction.
	"""
#	#To avoid doing anything twice. (odeint tends to do that.)
#	#---------------------------------------------------------
#	novel, result = self.check_novelty(t,psi)
#	if not novel:
#	    if self.my_id == 0:
#		print "Time: %2.2f / %2.2f au. Runtime: %2.2f---"%(
#		    t, self.total_duration, (time.time() - self.t_0)/60.)
#		self.debug_norm(t, psi, result)	
#		
#	    return result
#	##########################################################

	#Making a complex array. 
	psi_complex = psi[:len(psi)/2] + 1j * psi[len(psi)/2:] 
	
	dp_dt_complex = zeros(psi_complex.shape, dtype = complex)
	dp_dt_buffer= zeros(psi_complex.shape, dtype = complex)
	

	#Do operations.
	mat_vec = self.mat_vec_product(psi_complex, t)

	dp_dt_complex[self.my_slice] = self.solve_overlap(-1j * mat_vec)
	


	#Add and redistribute.
	dp_dt_complex = pypar.reduce(dp_dt_complex, pypar.SUM, 0, buffer = dp_dt_buffer)
	dp_dt_buffer = dp_dt_complex.copy()
	dp_dt_complex = pypar.broadcast(dp_dt_buffer, 0)
	


	#Making a float array.
	dp_dt = r_[real(dp_dt_buffer), imag(dp_dt_buffer)] 
	
	if self.my_id == 0:
	    print "Time: %2.2f / %2.2f au. Runtime: %2.2f"%(
		t, self.total_duration, (time.time() - self.t_0)/60.)
	    self.debug_norm(t, psi, dp_dt)	
	
	#Store latest result. ----------------------------------
	self.prev_out = dp_dt
	############################3###########################3
	return dp_dt

    def mat_vec_product(self, psi, t):
	"""
	psi_final = mat_vec_product(psi, t)

	Does the matrix vector multiplication with the Hamiltonian.

	Parameters
	----------
	psi : 1D complex array. Wavefunction.
	t : float. Time.

	Returns
	-------
	psi_final : 1D complex array. Result of the multiplication.

	Notes
	-----
	In the present form, one assumes real couplings, i.e. symmetric H.
	"""
	x = zeros(self.vib_basis_size * len(self.my_tasks), dtype = complex)

	#Matrix vector product.
	for i, j in enumerate(self.my_tasks):
	    slice_x = slice(i * self.vib_basis_size, (i + 1) * self.vib_basis_size)
	    slice_psi = slice(j * self.vib_basis_size, (j + 1) * self.vib_basis_size)
	    
	    x[slice_x]  = dot(self.h_0[:,:,i], psi[slice_psi])
	
	y = dot(self.h_1, psi)

	#Weigh with field strength, and add components.
	psi_final = x + self.time_function(t) * y
	
	return psi_final


    def time_function(self, t):
	"""
	field_strength = time_function(t)

	Returns the electrical field strength at the time <t>.
	Assumes an electrical field on the form 
	E(t) = sin**2(pi * t / T) * cos(omega * t)

	Parameters
	----------
	t : float. A time.

	Returns
	-------
	field_strength : float. Electrical field strength at the time <t>.
	"""

	if type(t) == float:
	    if t > self.pulse_duration:
		field_strength = 0.0
	    else:
		field_strength = (self.amplitude * 
		    sin(pi * t / self.pulse_duration)**2 *  
		    cos(self.omega * t))
	else:
	    field_strength = zeros(shape(t))
	    for i, time in enumerate(t):
		if time > self.pulse_duration:
		    temp_field_strength = 0.0
		else:
		    temp_field_strength = (self.amplitude * 
			sin(pi * time / self.pulse_duration)**2 *  
			cos(self.omega * time))

		field_strength[i] = temp_field_strength
	
	return field_strength



    def retrieve_hamiltonian(self, file_handle):
	"""
	hamilton_slice = retrieve_hamiltonian(file_handle)

	Retrieves this processors share of the hamiltonian.
	This is not entirely trivial, since only half the hamiltonian 
	is stored.
	
	Parameters
	----------
	file_handle : handle to HDF5 file where the time dependent 
	    hamiltonian is stored.
	
	Returns
	-------
	hamilton_slice : 2D complex array, this processors share of 
	    the hamiltonian.
	"""
	
	#WHEN THE ENTIRE MATRIX IS STORED.
	hamilton_slice = file_handle.root.couplings[
	    self.my_tasks[0] * self.vib_basis_size:
	    (self.my_tasks[-1] + 1) * self.vib_basis_size, :]


#	WHEN ONLY HALF THE MATRIX IS STORED:
#	hamilton_slice = zeros([len(self.my_tasks) * self.vib_basis_size, 
#	    self.basis_size], dtype = complex)
#	#Fist index will be different in local and stored hamiltonian.
#	for i_local, i_disc in enumerate(self.my_tasks):
#	    slice_1_local = slice(i_local * self.vib_basis_size, 
#		(i_local + 1) * self.vib_basis_size)
#	    slice_1_disc = slice(i_disc * self.vib_basis_size, 
#		(i_disc + 1) * self.vib_basis_size)
#
#	    for j in range(self.el_basis_size):
#		slice_2 = slice(j * self.vib_basis_size, 
#		    (j + 1) * self.vib_basis_size)
#
#		if j <= i_disc:
#		    hamilton_slice[slice_1_local, slice_2] = file_handle.root.couplings[
#			slice_1_disc, slice_2]
#		else:
#		    hamilton_slice[slice_1_local, slice_2] = transpose(
#			file_handle.root.couplings[slice_2, slice_1_disc])
#
	return hamilton_slice

    def solve_overlap(self, b):
	"""
	x = solve_overlap(b)

	Solve for the overlap matrix: S x = b.

	Parameters
	----------
	b : 1D complex array.
	
	Returns
	-------
	x : 1D complex array.
	"""
	x = zeros(b.shape, dtype = complex)
	
	#Solve for the B-spline overlap matrix.
	for i in range(len(self.my_tasks)): 
	    sub_slice = slice(i * self.vib_basis_size, 
		(i + 1) * self.vib_basis_size)
	    x[sub_slice] = lu_solve(self.overlap_fact, b[sub_slice])
	
	return x
	

