from scipy.linalg import lu_factor, lu_solve
from numpy import zeros, pi, sin, cos, dot, r_, real, imag, transpose

import time
import tables

class Ode_function_call:
    """
    This class provides a function that can serve as input to 
    scipy.integrate.odeint().
    """
    
    def __init__(self, filename_coupling, filename_vib, amplitude, omega, cycles):
	"""
	Ode_function_call(filename_coupling, filename_vib)

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
	self.pulse_duration = 2* pi /(self.omega) * self.cycles
	
	#TODO remove
	self.t_0 = time.time()


	#Open files
	self.f = tables.openFile(filename_vib)
	self.g = tables.openFile(filename_coupling)
	
	#Find the cholesky factorization of the overlap matrix.
	self.overlap_fact = lu_factor(self.f.root.overlap[:])
	self.overlap = self.f.root.overlap[:]#TODO remove
	
	#Name of electronic coupling file.
	filename_el = self.f.root.electronicFilename[0]
	
	#Retrieve index array.
	h = tables.openFile("../" + filename_el)
	self.index_array = h.root.index_array[:]
	h.close()

	#Basis sizes.
	self.el_basis_size = len(self.index_array)
	self.vib_basis_size = self.f.root.hamiltonian.shape[0]
	self.basis_size = self.el_basis_size * self.vib_basis_size

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
	#Making a complex array. 
	psi_complex = psi[:len(psi)/2] + 1j * psi[len(psi)/2:] 
	
	#Do operations.
	mat_vec = self.mat_vec_product(psi_complex, t)
	dp_dt_complex = self.solve_overlap(-1j * mat_vec)
	
	

	#Making a float array.
	dp_dt = r_[real(dp_dt_complex), imag(dp_dt_complex)] 
	
	print "Time: %2.2f / %2.2f au, runtime: %02.2f min"%(t, self.pulse_duration, (time.time() - self.t_0)/60. )
	
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
	
	#Time independent part.
	x = zeros(self.basis_size, dtype=complex)
	#Time dependent part.
	y = zeros(self.basis_size, dtype=complex)
	

	for i in range(self.el_basis_size):	    
	    #Indices of a submatrix.
	    slice_1 = slice(i*self.vib_basis_size, (i+1)*self.vib_basis_size)
	    
	    #Multiply with the time independent hamiltonian.
	    x[slice_1] = dot(self.f.root.hamiltonian[:,:,i], psi[slice_1])
	
	    for j in range(self.el_basis_size):
		if (self.index_array[i,1] + self.index_array[j,1])%2 == 0:
		    continue
		
		slice_2 = slice(j*self.vib_basis_size, 
		    (j+1)*self.vib_basis_size)
		
		if j <= i:
		    y[slice_1] += dot(self.g.root.couplings[slice_1, slice_2], 
			psi[slice_2])
		
		#If in the unwritten part of the array. 
		#TODO if couplings are complex, there will be trouble.
		else:
		    y[slice_1] += dot(transpose(self.g.root.couplings[slice_2, slice_1]), 
			psi[slice_2])
	
	#Weigh with field strength, and add components.
	psi_final =  x + self.time_function(t) * y
	
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
	if t > self.pulse_duration:
	    field_strength = 0.0
	else:
	    field_strength = (self.amplitude * 
		sin(pi * t / self.pulse_duration)**2 *  
		cos(self.omega * t))
	
	return field_strength


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
	x = zeros(self.basis_size, dtype=complex)
	
	for i in range(self.el_basis_size):
	    
	    #Indices of a submatrix.
	    my_slice = slice(i*self.vib_basis_size, (i+1)*self.vib_basis_size)
	    
	    #Solve for the B-spline overlap matrix.
	    x[my_slice] = lu_solve(self.overlap_fact, b[my_slice])
	
	return x

