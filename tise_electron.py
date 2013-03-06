from numpy import zeros, argsort, real, sqrt, outer, sum, ones
from numpy import conj, linspace, diff, argmax, shape
from pylab import find



import scipy
import psc_basis
import name_generator as name_gen
import tables


class TISE_electron:
    """
    Class for solving the electronic time independent Schoedinger 
    equation, TISE, for H2+. A prolate spheroidal coordinate system is used.
    This class sets up the matrices, S & H, solves the eigenvalue problem,
    and stores the data. Much of the theoretical foundation for this program 
    can be found in the article
    Kamta and Bandrauk, Phys. Rev. A 71, 053407 (2005).
    """
    def __init__(self, conf = None, filename = None, m = None, nu = None, mu = None, 
	    R = None, beta = None, theta = None):
	"""
	TISE_electron(filename = None, m = None, nu = None, mu = None, 
	    R = None, beta = None, theta = None)

	Constructor. Sets up the TISE instance.
	
	Parameters
	----------
	    
	    Option I
	    --------
	    conf : Config instance, defining the problem.

	    Option II
	    --------
	    filename : string, name of HDF5 file, 
		in which a config instance is saved.
	    
	    Option III
	    ---------
	    m : integer, maximal value of the m quantum number.	
	    mu : integer, maximal value of the mu quantum number.	
	    nu : integer, maximal value of the nu quantum number.	
	    R : float, internuclear distance.	
	    beta : float, modulus of the complex scaling.	
	    theta : float, argument of the complex scaling.	
	"""
	
	#Setting up the basis instace.
	self.basis = psc_basis.PSC_basis(conf = conf, filename = filename, 
		m = m, nu = nu, mu = mu, R = R, beta = beta, theta = theta)

	#Initialize hamiltonian matrix variable.
	self.hamiltonian = None

    def solve(self):
	"""
	E, V = solve()
	
	Solves the generalized eigenvalue problem Hc = eSc, 
	where H is the hamiltonian, S is the overlap matrix 
	and e is the eigenvalue. 
	Returns the eigenvalues, E, and eigenvectors, V.    

	Returns
	-------
	E : 1D float array, containing the sorted eigenvalues.
	V : 2D complex array, containing the normalized eigenvectors.
	"""

	#Make sure the matrices are made.
	if self.hamiltonian == None :
	    self.setup_hamiltonian()
	
	if self.basis.overlap_matrix == None :
	    self.basis.setup_overlap()


	#Finding the eigenvalues and eigenvectors.
	E, V = scipy.linalg.eig(self.hamiltonian, b=self.basis.overlap_matrix)
	
	#Sorting them according to rising eigenvalues.
	I = argsort(E)
	E = real(E[I])
	V = V[:,I]
	
	#Normalizing.
	V = V / self.normalization_factor(V)
	
	return E, V

    def save_eigenfunctions(self, E, V):
	"""
	save_eigenfunctions(E, V)

	Saves the eigenfunctions and eigenvalues to an HDF5 file, sorted by
	m and q quantum number.
	
	Parameters
	----------
	E : 1D float array, containing the sorted eigenvalues.
	V : 2D complex array, containing the normalized eigenvectors.

	Notes
	-----
	This method can be used for stand alone evaluation of the 
	electronic H2+ problem. If one solves a part of a 
	Born Oppenheimer problem, meaning it must be evaluated for many 
	different R (internuclear distances), use the method 
	save_eigenfunctions_R().
	"""
	#Construct an appropriate filename.
	filename = name_gen.electronic_eigenstates(self.basis.config)
	
    	#Organizing with regards to the quantum numbers.
	E_and_V_dictionary = self.sort_in_m_and_q(E,V)

	#Open an HDF5 file.
	f = tables.openFile(filename, 'w')

	#The file is organized in this way:
	#/
	#   config
	#   m/
	#	q/
	#	    E
	#	    V
	try:
	    for m_key, m_value in E_and_V_dictionary.iteritems():
		f.createGroup("/", m_key)
		for q_key, q_value in m_value.iteritems():
		    f.createGroup("/%s/"%m_key, q_key)
		    f.createArray("/%s/%s"%(m_key, q_key), "E", q_value[0])
		    f.createArray("/%s/%s"%(m_key, q_key), "V", q_value[1])
	
	finally:
	    f.close()
	
	#Saving the config info.
	self.basis.config.save_config(filename)
    
    
    def save_eigenfunctions_R(self, E, V, R):
	"""
	save_eigenfunctions_R(E, V, R)

	Saves the eigenfunctions and eigenvalues to an HDF5 file, sorted by
	m and q quantum number. Used when evaluating for several R. 
	
	Parameters
	----------
	E : 1D float array, containing the sorted eigenvalues.
	V : 2D complex array, containing the normalized eigenvectors.
	R : float, the internuclear distance in this case.

	Notes
	-----
	Assumes that the file is already made, the arrays are initialized,
	and the basis instance and the R vector are stored.

	The file is initialized from the assumption that the eigenstates are 
	equally distributed among the q values. Occasionaly this is not the 
	case, due to either our q-finding method, numerical inaccuracies, or
	perhaps the problem itself. This is most common when mu_max < 15. 
	When this occurs, it leads to an error during runtime.
	"""
	#Construct an appropriate filename.
	filename = name_gen.electronic_eigenstates_R(self.basis.config)
	
    	#Organizing with regards to the quantum numbers.
	E_and_V_dictionary = self.sort_in_m_and_q(E,V)

	#Open an HDF5 file, provided it exists.     
	f = tables.openFile(filename, 'r+')
	try:
	    R_grid = f.root.R_grid[:]

	    #Find index of current R in the R_grid.
	    R_index = find(R_grid == self.basis.config.R)[0]
	    
	    #The file is organized in this way:
	    #/
	    #	R_grid
	    #	config
	    #   m/
	    #	    q/
	    #		E
	    #		V
	    for m_key, m_value in E_and_V_dictionary.iteritems():
		for q_key, q_value in m_value.iteritems():
		    #Storing E & V at the right place in the arrays.
		    E_handle = eval("f.root.%s.%s.E"%(m_key, q_key))
		    E_handle[:,R_index] = E_and_V_dictionary[m_key][q_key][0]
		    
		    V_handle = eval("f.root.%s.%s.V"%(m_key, q_key))
		    V_handle[:,:,R_index] =E_and_V_dictionary[m_key][q_key][1]		
	finally:
	    f.close()

    def sort_in_m_and_q(self, E, V):
	"""
	E_and_V_dictionary = sort_in_m_and_q(E, V)

	Sorts the eigenfunctions according to the m and q quantum number, 
	(i.e. number of nodes in the eta part of the function 
	(polar angle function, roughly)). Returns a dictionary with m and q 
	values as keys.
	
	Parameters
	----------
	E : 1D float array, containing the sorted eigenvalues.
	V : 2D complex array, containing the normalized eigenvectors.
	
	Returns
	-------
	E_and_V_dictionary : nested dictionary instance, with m and q numbers as keys,
	    and a tuple of corresponding E and V constituents as values.

	Notes
	-----
	The q quantum number seems to correspond to the l quantum number in 
	atoms. It is explained further in 
	Barmaki et al., Phys. Rev. A 69, 043403 (2004). 
	"""
	
	#Initialize dictionaries.
	index_dictionary = {}
	E_and_V_dictionary = {}

	#Getting the q quantum numbers from the eigenfunction.
	q = self.extract_q(V)
	
	#Loop over eigenstates.
	for i in range(V.shape[1]):
	    #Singling out a wavefunction.
	    v = V[:,i]
	    
	    #Getting the m quantum number from the eigenfunction.
	    m = self.extract_m(v)
	    
	    m_key = name_gen.m_name(m) 
	    q_key = name_gen.q_name(q[i]) 
	    
	    #Add the index to the index dictionary.
	    if m_key in index_dictionary:
		if q_key in index_dictionary[m_key]:
		    index_dictionary[m_key][q_key].append(i)
		else:
		    index_dictionary[m_key][q_key] = [i]
	    else:
		index_dictionary[m_key] = {q_key:[i]}
	
	#Putting the actual eigenstates/energies in the dictionary.
	for m_key, m_value in index_dictionary.iteritems():
	    for q_key, indices in m_value.iteritems():
		
		if m_key in E_and_V_dictionary:
		    E_and_V_dictionary[m_key][q_key] = (E[indices], 
			V[:, indices])
		else:
		    E_and_V_dictionary[m_key] = {q_key:(E[indices], 
			V[:, indices])}
		    
	return E_and_V_dictionary


    def extract_m(self, v):
	"""
	m = extract_m(v):
	
	Discover the m quantum number of the eigenstate v.

	Parameters
	----------
	v : 1D complex array, an eigenvector.

	Returns
	-------
	m : integer, the m quantum number.
	"""
	#Index of the basis function with the largest contribution.
	max_index = argmax(abs(v))
		
	#Extracting the m quantum number.
	m, nu, mu = self.basis.index_iterator[max_index]

	return m

    def extract_q(self, V):
	"""
	q = extract_q(V):
	
	Discover the q quantum numbers of the eigenstates V.

	Parameters
	----------
	V : 2D complex array, eigenvectors.

	Returns
	-------
	q : 1D integer array, the q quantum numbers. (See Barmaki2004.)

	Notes
	-----
	This method has trouble if mu_max is below ~15. The trouble may be 
	inherent in the problem. 

	I don't know if this is a valid approach for m different from 0.
	I have assumed the end points (+-1) will not count as nodes, but I
	cannot cite anything clever.
	"""
	eta_grid = linspace(-1,1,1000)

	if self.basis.config.m_max == 0:
	    pass
	else:
	    #Avoiding the problem that eta_fun has nodes in +-1 for |m| > 0.
	    eta_grid = eta_grid[1:-1]
	
	#Number of eigenstates.
	nr_used = V.shape[1]
	
	#Initializing angular functions.
	eta_funs = zeros([len(eta_grid), nr_used, self.basis.config.nu_max+1])
	eta_fun = zeros([len(eta_grid), nr_used])
	
	#Add up the angular parts of the wavefunctions. (For each nu.)
	for i, [m, nu, mu] in enumerate(self.basis.index_iterator):
	    eta_funs[:,:, nu - m] += outer(
		self.basis.evaluate_V(eta_grid, m, mu), V[i])
	
	#Chooses the nu with the most population, to avoid that it is near 0.
	for i in range(nr_used):
	    largest_contribution = argmax(sum(abs(eta_funs[:,i,:]),axis=0))
	    eta_fun[:,i] = eta_funs[:, i, largest_contribution]

	#Initialize return array.
	q = zeros(nr_used, dtype = int)
	
	#Boolean array, giving the sign.
	sign_eta_fun = eta_fun < 0
	
	#Array of changes of sign.
	sign_changes = diff(sign_eta_fun, axis=0)

	#Number of sign changes/roots/nodes.
	q = sum(abs(sign_changes) > 0.5, axis=0)

	return q
    
    def align_all_phases(self):
	"""
	align_all_phases()

	Makes sure the eigenfunctions saved to file does not change sign from
	one R to another.
	"""
	#File where the eigenstates are stored.
	filename = name_gen.electronic_eigenstates_R(self.basis.config)
	
	m_max = self.basis.config.m_max
	
	#q_max is the same as mu_max.
	q_max = self.basis.config.mu_max
	
	f = tables.openFile(filename, "r+")

	try:
	    for m in range(-m_max, m_max + 1):
		m_group = name_gen.m_name(m)
		for q in range(q_max + 1):
		    q_group = name_gen.q_name(q)
		    V = eval("f.root.%s.%s.V"%(m_group, q_group))
		    #Flipping the appropriate wavefunctions.
		    self.align_phases(V)

	finally:
	    f.close()
    

    def align_phases(self, V):
	"""
	align_phases(V)
	
	Makes sure the saved eigenfunctions <V> does not changed sign
	all over the place.

	Parameters
	----------
	V : file handle to 3D complex array, (eignefunctions), 
	    in the HDF5 file.
	"""

	#Adding and subtracting the eigenfunctions at neighbouring R values.
	sum_neighbours = V[:,:,:-1] + V[:,:,1:]
	diff_neighbours = V[:,:,:-1] - V[:,:,1:]
	
	#Adding the components of the sum/difference wavefunctions.
	sum_neighbours  = sum(abs(sum_neighbours), axis = 0)
	diff_neighbours = sum(abs(diff_neighbours), axis = 0)
	
	#Finds out if the sum of the neighbouring wavefunctions are bigger than
	#the difference. (Signifying a filpped wavefunction.)
	compare_sum_and_diff = zeros(
	    [sum_neighbours.shape[0], sum_neighbours.shape[1], 2])
	
	compare_sum_and_diff[:,:,0] = sum_neighbours
	compare_sum_and_diff[:,:,1] = diff_neighbours
	
	compare_sum_and_diff = argmax(compare_sum_and_diff, axis = 2)

	for i in range(compare_sum_and_diff.shape[1]):
	    #How many times have the wavefunction flipped?
	    nr_of_changes = sum(compare_sum_and_diff[:,:i + 1], axis = 1)
	    
	    #Flips back by multiplying with -1.
	    signs = (-1 * ones(shape(nr_of_changes))) ** nr_of_changes

	    V[:,:,i + 1] *= signs


    def normalization_factor(self, V):
	"""
	N = normalizaton_factor(V)

	Calculates the inner product <fun,fun>, for all eigenfunctions.
	Simply divide the eigenfunctions by the factors N.

	Parameters
	----------
	V : 2D array containing the unnormalized eigenvectors.

	Returns
	-------
	N : Vector containing the normalization factors.

	Examples
	--------
	>>> E, V = scipy.linalg.eig(self.H, b=self.S)
	>>> V = V/tise.normalisation_factor(V)
	"""
	
	#Number of diagonalized states that are kept.
	nr_used = V.shape[1]

	#Normalization constants.
	N = zeros([1,nr_used])
	
	#int(|psi|**2) = sum_ij c_i* c_j S_ij.
	for i, fill_1 in enumerate(self.basis.index_iterator): 
	    for j, fill_2 in enumerate(self.basis.index_iterator): 
		N += conj(V[i]) * V[j] * self.basis.overlap_matrix[i,j]
	
	return  sqrt(N) 



    def setup_hamiltonian(self):
	"""
	setup_hamiltonian()

	Calculates matrix elements for the hamiltonian matrix,
	and adds the matrix as a class variable.
	The matrix elements are described in Appendix A in Kamta2005.
	Eq. (A3) gives the expression for a general hamiltonian matrix element.
	"""
	#Initialize hamiltonian matrix.
	hamiltonian = zeros([self.basis.basis_size, self.basis.basis_size], 
		dtype = complex)
	
	#Looping over indices.
	#<bra|
	for i, [m_prime,nu_prime,mu_prime] in enumerate(
	    self.basis.index_iterator):
	    #|ket>
	    for j, [m, nu, mu] in enumerate(
		self.basis.index_iterator):
		#Selection rule.
		if m_prime == m and mu_prime == mu:
		    #Upper triangular part of the matrix.
		    if j >= i:
			hamiltonian[i,j] = -self.basis.config.R/4.0 * (
			      self.find_h(m, nu_prime, nu) 
			    * self.basis.find_d_tilde(0, m, mu_prime, mu)
			    + self.basis.find_d(0, m, nu_prime, nu) 
			    * self.find_h_tilde(m, mu_prime, mu))
		    else:
			#Lower triangular part is equal to the upper part.
			#TODO Should this be the conjugated? Yes?? No??
			hamiltonian[i,j] = hamiltonian[j,i]
	
	#Making the matrix a class/instance variable.
	self.hamiltonian = hamiltonian
	

    def find_h(self, m, nu_prime, nu):
	"""
	h = find_h(m, nu_prime, nu)

	Calculates the "h" part of the hamiltonian matrix element, as 
	described in Kamta2005, equation (A6). Details on how the equation is 
	transformed into an implemantable expression are given in the file 
	Documentation/A_6.pdf.

	Parameters
	----------
	m : int, quantum number, corresponding to the electron's 
	    angular momentum projection onto the z axis.
	nu_prime : int, 'xi' quantum number for the <bra| basis function.
	nu : int, 'xi' quantum number for the |ket> basis function.
	
	Returns
	-------
	h : complex, matrix element of the hamiltonian.
	"""
	#Necessary functions and factors.
	#--------
	#Quadrature nodes.
	X = self.basis.quadrature_object.nodes
	
	#Basis parameters, for convenience.
	alpha = self.basis.config.alpha
	R = self.basis.config.R

	#Nuclear charge.
	Z = 1.0

	#Normalization for the <bra| basis function.
	N_prime = self.basis.find_N(m, nu_prime)
	
	#<bra| basis function, sans normalization and exponential function.
	U_prime = self.U_integrand(m, nu_prime)

	#Normalization for the |ket> basis function.
	N = self.basis.find_N(m, nu)

	#|ket> basis function, sans normalization and exponential function.
	U = self.U_integrand(m, nu)

	#Differentiated |ket> basis function, sans etc...
	dU = self.dU_integrand(m, nu)
	
	#Twice differentiated |ket> basis function, sans etc...
	d2U = self.d2U_integrand(m, nu)
	#-----------
	
	#The integrand. (Not the exponential part.)
	integrand = U_prime * (
	      ((X / alpha) + 2.0) * dU 
	    + (X / (2 * alpha)) * (X / (2 * alpha) + 2) * d2U 
	    + (2 * Z * R * (X / (2 * alpha) + 1) 
	    - 4 * alpha**2 * m**2 / (X * (X + 4 * alpha)))*U)
	
	#Integrate with quadrature formula.
	h_integral = self.basis.quadrature_object.integrate(integrand)
	
	#The h part of the matrix element.
	h = N_prime * N / (2 * alpha) * h_integral
	
	return h


    def find_h_tilde(self, m, mu_prime, mu):
	"""
	h_tilde = find_h_tilde(self, m, mu_prime, mu)

	Calculates the h_tilde part of a matrix element, as described in 
	Kamta2005, eq. (A7) and eq. (A14).

	Parameters
	----------
	m : int, quantum number, corresponding to the electron's 
	    angular momentum projection onto the z axis.
	mu_prime : int, 'eta' quantum number for the <bra| basis function.
	mu : int, 'eta' quantum number for the |ket> basis function.

	Returns
	-------
	h_tilde : int, eta integral part of matrix element.
	"""
	#Equation (A14) in Kamta2005.
	h_tilde = -mu * (mu + 1) * (mu_prime == mu)

	return h_tilde

    def U_integrand(self, m, nu):
	"""
	U = U_integrand(m, nu)

	Evaluates a U basis function (see Notes) in the quadrature rule nodes.
	See reasoning in the file "Documentation/A_6.pdf".
	
	Parameters
	----------
	m : integer, the angular momentum projection quantum number.
	nu : integer, some sort of quantum number associated with U.

	Returns
	-------
	U : 1D complex array, the result of the evaluation. 
	
	Notes
	-----
	The normalization factor (N) and exponential function are 
	factored out of the basis function. 
	They are taken care of in "find_h()".
	"""
	#Nodes of the quadrature formula.
	X = self.basis.quadrature_object.nodes
	
	#Laguerre polynomial, evaluated in X.
	L = self.basis.get_laguerre(nu - abs(m), 2 * abs(m))

	alpha = self.basis.config.alpha
	
	#Selected parts of the U basis function.	
	U = ((X / (2.0 * alpha)) * (X / (2.0 * alpha) + 2))**(abs(m)/2.0) * L
	
	return U
	
    
    def dU_integrand(self, m, nu):
	"""
	dU = dU_integrand(m, nu)

	Evaluates the differentiation of a U basis function (see Notes) 
	in the quadrature rule nodes.
	See reasoning in the file "Documentation/A_6.pdf".
	
	Parameters
	----------
	m : integer, the angular momentum projection quantum number.
	nu : integer, some sort of quantum number associated with U.

	Returns
	-------
	dU : 1D complex array, the result of the evaluation. 
	
	Notes
	-----
	The normalization factor (N) and exponential function are 
	factored out of the basis function. 
	They are taken care of in "find_h()".
	"""
	#Nodes of the quadrature formula.
	X = self.basis.quadrature_object.nodes
	
	alpha = self.basis.config.alpha
	
	#Absolute value of m is used throughout.
	m = abs(m)

	#Laguerre polynomial, evaluated in X.
	L_0 = self.basis.get_laguerre(nu - m + 0, 2 * m)
	L_1 = self.basis.get_laguerre(nu - m + 1, 2 * m)

	
	#Selected parts of the differentiated U basis function. 	
	dU = (X / ( 2 * alpha) * (X / (2 * alpha) + 2))**(m/2.) * (
	    (alpha - 4 * alpha**2 * m / (X * (X + 4 * alpha)) 
	    - 2 * alpha * (nu + 1.) / X) * L_0 
	    + 2 * alpha * (nu - m + 1)/X * L_1)	

	return dU
	
    def d2U_integrand(self, m, nu):
	"""
	d2U = d2U_integrand(m, nu)

	Evaluates the 2nd differentiation of a U basis function (see Notes) 
	in the quadrature rule nodes.
	See reasoning in the file "Documentation/A_6.pdf".
	
	Parameters
	----------
	m : integer, the angular momentum projection quantum number.
	nu : integer, some sort of quantum number associated with U.

	Returns
	-------
	d2U : 1D complex array, the result of the evaluation. 
	
	Notes
	-----
	The normalization factor (N) and exponential function are 
	factored out of the basis function. 
	They are taken care of in "find_h()".
	"""

	#Nodes of the quadrature formula.
	X = self.basis.quadrature_object.nodes
	
	alpha = self.basis.config.alpha
	
	#Absolute value of m is used throughout.
	m = abs(m)

	#Laguerre polynomial, evaluated in X.
	L_0 = self.basis.get_laguerre(nu - m + 0, 2 * m)
	L_1 = self.basis.get_laguerre(nu - m + 1, 2 * m)
	L_2 = self.basis.get_laguerre(nu - m + 2, 2 * m)

	
	#Selected parts of the 2nd differentiated U basis function.	
	d2U = (X / (2 * alpha) * (X / (2 * alpha) + 2))**(m/2.) * (
	    (  
		alpha**2
		+ 2 * alpha**2 * m / (X + 4 * alpha)
		- 2 * alpha**2 * (2 * nu + m + 2) / X
		+ 16 * alpha**4 * m * (m - 2)/ (X**2 * (X + 4 * alpha)**2) 
		+ 4 * alpha**2 * (nu + 1) * (nu + 2) / X**2
		+ 16 * alpha**3 * m * (nu + 2) / (X**2 * (X + 4 * alpha))
	    ) * L_0 
	    +
	    (
		4 * alpha * (nu - m + 1) * ( 
		  alpha / X
		- 2 * alpha * (nu + 2) / X**2
		- 4 * alpha**2 * m / (X**2 * (X + 4 * alpha)))
	    ) * L_1
	    + 
	    (
		4 * alpha**2 * (nu - m + 1) * (nu - m + 2) / X**2	
	    ) * L_2
	    )

	return d2U
    

