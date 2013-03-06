from numpy import max, min, argsort, zeros, sqrt, shape, reshape, real

import gausslegendrequadrature as quad
import bsplines
import scipy.linalg
import scipy.interpolate
import tables

class Bspline_basis:
    """
    This class contains most methods needed for solving the Schroedinger 
    equation in 1D using a B-spline basis. Its purpose is to solve the 
    vibrational part of the H2+ Born Oppenheimer problem.
    """

    def __init__(self, xmin, xmax, xsize, order):
	"""
	"""
	#Create Bspline instance.
	self.bsplines = bsplines.Bspline(xmin, xmax, xsize, order)
	self.spline_order = order
	self.nr_splines = self.bsplines.number_of_bsplines

	#Create quadrature instance.
	self.quadrature = quad.Gauss_legendre_quadrature_rule(order)
	self.quad_order = self.quadrature.rule_order

	#Tabulate the B-splines.
	spline_table, x = self.bsplines.make_bspline_table()
	self.spline_table = spline_table
	self.x_values = x
	
	#Initialize the matrix varibles.
	self.kinetic_hamiltonian = None
	self.overlap_matrix = None


    def setup_kinetic_hamiltonian(self):
	"""
	setup_kinetic_hamiltonian()

	Calculates the kinetic hamiltonian matrix of the B-spline basis.

	H_ij = <B_i | -1/M * d2/dR2 | B_j>
	     = 1/M  <dB_i | dB_j>
	"""
	#Initialize kinetic hamiltonian matrix.
	kinetic_matrix = zeros([self.nr_splines, self.nr_splines])

	#Loop over elements.
	for i in range(self.nr_splines):
	    for j in range(self.nr_splines):
		#Exploiting the symmetrical nature of the matrix.
		if i <= j:
		    kinetic_matrix[i,j] = self.integration(i,j,
			differentiation = True)
		else:
		    kinetic_matrix[i,j] = kinetic_matrix[j,i]
	
	#Mass of a nucleus.
	mass = 1836.15266

	#Updating the class variable.
	self.kinetic_hamiltonian = -1 * kinetic_matrix/mass


    def setup_overlap_matrix(self):
	"""
	setup_overlap_matrix()

	Calculates the overlap matrix of the B-spline basis.
	"""

	#Initializes overlap matrix.
	overlap = zeros([self.nr_splines, self.nr_splines])

	#Loops over elements.
	for i in range(self.nr_splines):
	    for j in range(self.nr_splines):
		#Exploiting the symmetrical nature of the matrix.
		if i <= j:
		    overlap[i,j] = self.integration(i,j)
		else:
		    overlap[i,j] = overlap[j,i]
	
	#Updating the class variable.
	self.overlap_matrix = overlap


    def setup_potential_matrix(self, r_grid, r_function):
	"""
	potential_matrix = setup_potential_matrix(r_grid, r_function)

	Sets up and returns the potential matrix. The potential is described by 
	<r_function>.
	
	Parameters
	----------
	r_grid : 1D float array.
	r_function : 1D float array, a function of <r_grid>.

	Returns
	-------
	potential_matrix : 2D float array, containing the potential energy 
	    part of the hamiltonian matrix.	
	"""
	
	#Cast the potential into the same format as x_values.
	function_array = self.cast_to_integration_format(r_grid, r_function)

	#Initializes overlap matrix.
	potential_matrix = zeros([self.nr_splines, self.nr_splines])

	#Loops over elements.
	for i in range(self.nr_splines):
	    for j in range(self.nr_splines):
		#Exploiting the symmetrical nature of the matrix.
		if i <= j:
		    potential_matrix[i,j] = self.integration(i, j, 
			operator = function_array)
		else:
		    potential_matrix[i,j] = potential_matrix[j,i]

	return potential_matrix



    def solve(self, H, nr_kept):
	"""
	E,V = solve(H, nr_kept)
	
	Solves the generalized eigenvalue problem Hc = eSc, where H is the hamiltonian,
	S is the overlap matrix (between B-splines) and e is the eigenvalue. 
	Returns the eigenvalues, E, and eigenvectors, V.    

	Parametres
	----------
	H : Hamiltonian matrix.
	nr_kept : Integer. The number of states to be kept.

	Returns
	-------
	E : Vector containing the sorted eigenvalues.
	V : Array containing the normalized eigenvectors.
	"""
	    
	if self.overlap_matrix == None:
	    self.setup_overlap_matrix()

	#Finding the eigenvalues and eigenvectors.
	E, V = scipy.linalg.eig(H, b=self.overlap_matrix)
	
	#Sorting them according to rising eigenvalues.
	I = argsort(E)
	E = E[I][:nr_kept]
	V = V[:,I][:,:nr_kept]
	
	#Normalizing.
	V = V/self.normalisation_factor(V)
	
	return E,V

    def normalisation_factor(self, V):
	"""
	N = normalizaton_factor(V)

	Calculates the inner product <fun,fun>, for all eigenfunctions.
	Simply divide the eigenfunctions by the factors N.

	Parametres
	----------
	V : 2D array containing the unnormalized eigenvectors.

	Returns
	-------
	N : Vector containing the normalization factors.

	Examples
	--------
	>>> E, V = scipy.linalg.eig(self.H, b=self.overlap_matrix)
	>>> V = V/self.normalisation_factor(V)
	"""

	#B-spline basis
	nr_used = V.shape[1]

	#Normalization constants
	N = zeros([1,nr_used])
	for i in range(self.nr_splines): 
	    for j in range(i,self.nr_splines):
		#Only the nonzero integrands
		if abs(i - j) < self.spline_order:    
		    #Exploiting the symmetry
		    if i == j:
			N += V[i] * V[j] * self.overlap_matrix[i,j]
		    else:
			N += 2 * V[i] * V[j] * self.overlap_matrix[i,j]
	return  sqrt(N) 


    
    def cast_to_integration_format(self, r_grid, r_function):
	"""
	output = cast_to_integration_format(r_grid, r_function)

	Transforms <r_function> - given on the <r_grid> - into the format of 
	the x_values. This will expedite the integration.

	Parameters
	----------
	r_grid : 1D float array.
	r_function : 1D float array, a function of <r_grid>.

	Returns
	-------
	output : 3D float array, <r_function> evaluated in x_values.
	"""

	#Warn if the grids don't match.
	r_max = max(r_grid)
	x_max = max(self.x_values.ravel())
	if x_max > r_max:
	    print "WARNING: function is not defined on the entire grid."

	#Create an interpolator for r_function on r_grid.
	interpolator = scipy.interpolate.UnivariateSpline(r_grid, r_function,
	    s = 0)

	#Evaluates r_function in x_values. 1D output.
	raw_output = interpolator(self.x_values.ravel())
	
	#Reshapes the output into the shape of x_values. 
	output = reshape(raw_output, shape(self.x_values), order = "C")

	return output

    def bspline_expansion(self, grid, f):
	"""
	c =  bspline_expansion(grid, f)

	Expand a function f in the B-spline basis. Since B-splines are not
	orthogonal, the usual projection method yields a linear system of
	equations, Sc = b, where S is the B-spline overlap matrix, c are the 
	expansion coefficients and b is the projection of f onto the basis.
	The matrix S is banded with 2k - 1 bands (k is B-spline order). 
	Inverting S yields the expansion coefficients, c = S'b.

	Parameters
	----------
	grid : 1d float array.
	f : 1d float array, a function defined on <grid>.

	Returns
	-------
	c : 1d float array, the expansion coefficients of <f>.
	"""
	#Cast the function to the format of the B-spline table.
	f_table = self.cast_to_integration_format(grid, f)
	
	#Projection vector		
	b = zeros(self.nr_splines)
	
	#Dummy spline evaluations.
	dummy = self.spline_table[0,:,:,0] * 0 + 1

	for j in range(self.nr_splines):
	    #Calculate projection < B_j | f >
	    b[j] += self.quadrature.integrate_from_table(
		self.spline_table[j,:,:,0], dummy, 
		self.x_values[j,:,:], fox = f_table[j,:,:])

	#Solve linear system of equations to obtain expansion coefficients
	c = scipy.linalg.solve(self.overlap_matrix, b)

	return c


    def integration(self, i, j, operator = None, differentiation = False):
	"""
	integral = integration(i, j, operator = None, differentiation = False)

	Calculates an integral involving 2 b-splines and possibly an operator.
	Uses table values for Bsplines and x values. 
	
	Parameters
	----------
	i : integer, index of one of the left B-spline. 
	j : integer, index of one of the right B-spline. 
	operator : 3D float array, containing the function values evaluated in
	    x_values. 
	differentiation : boolean, tells if this is an element of the kinetic
	    hamiltonian matrix, and the splines should be differentiated.

	Returns
	-------
	integral : float, the result of the integration.
	"""
		
	#Number of intervals that has nonzero values for spline i and j
	nr_overlap = self.spline_order - abs(i-j)

	#Eventual collection of function values.
	fox = None

	if nr_overlap > 0:
	    index_1 = min([i,j])
	    index_2 = max([i,j])
							
	    #Matrices to store the relevant spline- and x data
	    B1 = self.spline_table[index_1, -nr_overlap:, :, differentiation]
	    B2 = self.spline_table[index_2, :nr_overlap, :, differentiation]
	    x = self.x_values[index_2, :nr_overlap, :]
	    
	    if operator != None:
		#The function of x.
		fox = operator[index_2, :nr_overlap, :]
	    

	    integral = (-1)**differentiation \
		* self.quadrature.integrate_from_table(B1, B2, x, fox = fox)
	    
	    return integral
	
	else:
	    return 0


    def save_eigenstates(self, filename, E, V, H, current_index):
	"""
	save_eigenstates(filename, E, V, H, current_index)

	Saves the eigenvalues and eigenstates to a designated HDF5 file.
	The file must be properly initialized for this to work.

	Parameters
	----------
	filename : string, the name of the HDF5 file where the states 
	    should be stored.
	E : 1D float array, the eigenvalues/energies of the H2+ states.
	V : 2D float array, the eigenstates/wavefunctions of the vibrational 
	    H2+ states.
	H : 2D float array, the time independent vibrational 
	    hamiltonian matrix.
	current_index : integer, the index of the electronic state in the 
	    eigenstate basis, whose energies act as the potential for this 
	    vibrational problem.
	"""

	#Save eigenstates/energies to file.	
	f = tables.openFile(filename,'r+')
	try:
	    f.root.V[:, :, current_index] = real(V)
	    f.root.hamiltonian[:, :, current_index] = real(H)
	    f.root.E[:, current_index] = E
	finally:
	    f.close()
	


    def save_couplings(self, filename, couplings, i, j):
	"""
	save_couplings(filename, couplings, i, j)

	Saves the hamiltonian matrix elements to the designated HDF5 file.
	The file must be properly initialized for this to work.
	
	Parameters
	----------
	filename : string, the name of the HDF5 file where the states 
	    should be stored.
	couplings : 2D complex array, containing the H2+ matrix elements 
	    for one combination of electronic states.
	i : integer, the index of the <bra| electronic state.
	j : integer, the index of the |ket> electronic state.
	"""
	
	#Size of the coupling array.
	nr_couplings = len(couplings)

	#Indices.
	start_ind_i = nr_couplings * i
	end_ind_i = start_ind_i + nr_couplings
	
	start_ind_j = nr_couplings * j
	end_ind_j = start_ind_j + nr_couplings
	
	#Saving the couplings to file.
	f = tables.openFile(filename,'r+')
	try:
	    f.root.couplings[start_ind_i:end_ind_i, 
		start_ind_j:end_ind_j] = couplings

	finally:
	    f.close()



