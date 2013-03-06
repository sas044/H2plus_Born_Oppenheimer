#Useful Python methods.
from numpy import exp, pi, sqrt, zeros, real, outer, array, linspace, arctan2
from numpy import flipud, r_, transpose, log, conj
from pylab import figure, pcolormesh, xlabel, ylabel, title, show
from scipy.special import genlaguerre, lpmn
from scipy.interpolate import  RectBivariateSpline

#Supporting classes.
import config
import index_iterator
import gausslaguerrequadrature as quadrature




class PSC_basis:
    """
    This class contains useful methods when using prolate spheroidal 
    coordinates (PSC). The basis is described in 
    Kamta and Bandrauk, Phys. Rev. A 71, 053407 (2005). 
    """

    def __init__(self, conf = None, filename = None, m = None, nu = None, mu = None, 
	    R = None, beta = None, theta = None):
	"""
	PSC_basis(filename = None, m = None, nu = None, mu = None, 
	    R = None, beta = None, theta = None)

	Constructor, initiating the basis instance.

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
	#Instance defined from the representation parametres:
	    #m, nu, mu, alpha, beta, theta
	#These can be found in a config instance or 
	#a file with wavefunctions. 
        self.config = config.Config(conf = conf, filename = filename, 
		m = m, nu = nu, mu = mu, R = R, beta = beta, theta = theta)
	
	#Iterator for looping over basis functions.
	self.index_iterator = index_iterator.Index_iterator(self.config)
	
	#Number of basis functions.
	self.basis_size = len([i for i in self.index_iterator])
	
	#Integration variables.
	#Order of the quadrature formula.
	self.rule_order = (2 + self.config.m_max + self.config.nu_max)
	
	#Make Gauss Laguerre quadrature instance.
	self.quadrature_object = quadrature.Gauss_laguerre_quadrature_rule(
	    self.rule_order)
	
	#Sets up table of Laguerre polynomials.
	self.tabulate_laguerre()

	#Initialize overlap matrix variable.
	self.overlap_matrix = None
    
    def get_laguerre(self, degree, order):
	"""
	L = get_laguerre(degree, order)

	Retrieves the correct associated laguerre polynomial from the table,
	so you don't have to remember the clever way the table is organised.
	Typical notation for the polynomial is L_{degree}^{order}.

	Parameters
	----------
	degree : integer, degree of the polynomial, typically (nu - |m|).
	order : integer, order of the polynomial, typically (2|m|)

	Returns
	-------
	L : 1D float array, containing the polynomial evaluated in the 
	    quadrature nodes.
	"""
	try:
	    #Map order to index.
	    index_order = int(order/2)

	    #Retrieve polynomial.
	    L = self.laguerre_table[index_order, degree,:]

	except IndexError:
	    error_message = ("The degree or order does not correspond to a " +
		"tabulated polynomial.")
	    print error_message

	return L

    def evaluate_U(self, xi_grid, m, nu):
	"""
	result = evaluate_U(xi_grid, m, nu):

	Evaluates U on a grid. U is the basis function for the xi coordinate.
	U is defined in eq. (7) in Kamta2005.

	Parameters
	----------
	xi_grid : 1D float array, the xi values for which one wants to
	    evaluate U.
	m : integer, the angular momentum projection quantum number.
	nu : integer, some sort of quantum number associated with U.

	Returns
	-------
	result : 1D complex array, the result of the evaluation. 
	"""
	#Eq. (7) in Kamta2005.
	#---------------------
	
	alpha = self.config.alpha
	m = abs(m)

	#Normalization factor, N^m_nu.
	N = self.find_N(m, nu)
	
	#Exponential factor.
	exponential_factor = exp(-alpha * (xi_grid - 1))
	
	#Generalized/associated Laguerre polynomial. 
	#   L^a_b(x) => genlaguerre(b,a)(x)
	#   Warning! Not reliable results for higher orders.
	laguerre_factor = genlaguerre(nu - m, 2*m)(2*alpha * (xi_grid - 1))
	

	
	#All together now.
	result = N * exponential_factor * (xi_grid**2 - 1)**(m/2.) * laguerre_factor
	
	return result 


    def evaluate_V(self, eta_grid, m, mu):
	"""
	result = evaluate_V(eta_grid, m, mu)

	Evaluates V on a grid. V is the basis function for the eta coordinate.
	V is defined in eq. (8) in Kamta2005.
	
	Parameters
	----------
	eta_grid : 1D float array, the eta values for which one wants to
	    evaluate V.
	m : integer, the angular momentum projection quantum number.
	mu : integer, some sort of quantum number associated with V.

	Returns
	-------
	result : 1D complex array, the result of the evaluation. 

	"""
	#Eq. (8) in Kamta2005.

	#Normalization factor, M^m_mu.
	M =  self.find_M(m, mu)

	#Associated Legendre polynomial of the first kind. 
	#P^{order}_{degree} => lpmn(order, degree) 

	#Initalize array.
	legendre_factor = zeros(len(eta_grid), dtype=complex)

	#Function does not allow for array input.
	for i, eta in enumerate(eta_grid):
	    #Function returns all degrees and orders up to those given.
	    P, dP =  lpmn(m, mu, eta)
	    #Selects the correct polynomial.
	    legendre_factor[i] = P[-1,-1]
	    

	#All together now.
	result = M * legendre_factor
	
	return result
	

    def evaluate_W(self, phi_grid, m):
	"""
	result = evaluate_W(phi_grid, m)

	Evaluates W on a grid. W is the basis function for the phi coordinate.
	W is defined as W = exp(1j * m * phi) / sqrt(2 * pi)
	See Kamta2005, eq. (6). (The function is not called W, but the name 
	seemed a natural extrapolation of the existing naming convention.)
	
	Parameters
	----------
	phi_grid : 1D float array, the phi values for which one wants to
	    evaluate W.
	m : integer, the angular momentum projection quantum number.

	Returns
	-------
	result : 1D complex array, the result of the evaluation. 
	"""

	#Part of eq. (6) in Kamta2005.
	return exp(1.0j * m * phi_grid) / sqrt(2 * pi)



    def probability_on_PSC_grid(self, psi, xi_grid, eta_grid, phi_grid):
	"""
	probability = probaility_on_PSC_grid(psi, xi_grid, eta_grid, phi_grid)

	Finds the probability distribution on a 
	Prolate Spheroidal Coordinate (PSC) grid.

	Parameters
	-----------
	psi : 1D complex array, the wavefunction in the PSC basis.
	xi_grid : 1D float array, the xi values for which one wants to
	    evaluate U.	
	eta_grid : 1D float array, the eta values for which one wants to
	    evaluate V.
	phi_grid : 1D float array, the phi values for which one wants to
	    evaluate W.
	
	Returns
	-------
	probability : 3D float array, the probability on the defined 
	    xi/eta/phi grid.
	"""
	#Initializing wavefunction array.
	wavefunction = zeros([len(xi_grid), len(eta_grid), len(phi_grid)], 
		dtype = complex)
	
	#Initializing return array.
	probability = zeros([len(xi_grid), len(eta_grid), len(phi_grid)])

	
	#Looping over elements in the wavefunction.
	for i, [m, nu, mu] in enumerate(self.index_iterator):
	    temp_U = self.evaluate_U(xi_grid, m, nu)
	    temp_V = self.evaluate_V(eta_grid, m, mu)
	    
	    
	    #Outer product of U and V basis functions.
	    temp_UV = outer(temp_U, temp_V)
	    
	    #Looping over phi values.
	    for j, phi in enumerate(phi_grid):
		temp_W = self.evaluate_W(phi, m)
		#Add each contribution to the final wavefunction.
		wavefunction[:,:,j] += psi[i] * temp_UV * temp_W
	
	#Probability = absolute squared of the wavefunction.
	probability = real(abs(wavefunction)**2)

	#--------
	#Looking at wavefunction (instead of probability), 
	#for debugging purposes.
	#
	#probability = real(wavefunction)
	#--------

	return probability



    def setup_overlap(self):
	"""
	setup_overlap()

	Calculates matrix elements for the overlap matrix,
	and adds the overlap matrix as a class variable.
	The matrix elements are described in Appendix A in Kamta2005.
	Eq. (A2) gives the expression for a general overlap matrix element.
	"""
	#Initialize overlap matrix.
	overlap_matrix = zeros([self.basis_size, self.basis_size], 
		dtype = complex)
	
	#Looping over indices.
	#<bra|
	for i, [m_prime,nu_prime,mu_prime] in enumerate(self.index_iterator):
	    #|ket>
	    for j, [m, nu, mu] in enumerate(self.index_iterator):
		#Selection rule.
		if m_prime == m:
		    #Upper triangular part of the matrix.
		    if j >= i:
			overlap_matrix[i,j] = (self.config.R/2.)**3 * (
			      self.find_d(2, m, nu_prime, nu) 
			    * self.find_d_tilde(0, m, mu_prime, mu)
			    - self.find_d(0, m, nu_prime, nu) 
			    * self.find_d_tilde(2, m, mu_prime, mu))
		    else:
			#Lower triangular part is equal to the upper part.
			    #TODO Should this be the conjugated? Yes?? No??
			    #Might not be tht simple. Possibly (A6) & (A8) in
			    #Kamta2005 should be modified.
			overlap_matrix[i,j] = conj(overlap_matrix[j,i])
	
	#Making the matrix a class/instance valiable.
	self.overlap_matrix = overlap_matrix


    def find_d(self, q, m, nu_prime, nu):
	"""
	d = find_d(q, m, nu_prime, nu)
	
	Evaluates the 'd' part of the matrix element, as described in eq. (A4)
	and eq. (A8) in Kamta2005.

	Parameters
	----------
	q : int, typically 0 or 2, described in eq. (A4) in Kamta2005.
	m : int, quantum number, corresponding to the electron's 
	    angular momentum projection onto the z axis.
	nu_prime : int, 'xi' quantum number for the <bra| basis function.
	nu : int, 'xi' quantum number for the |ket> basis function.

	Returns
	-------
	d : complex, xi integral part of matrix element.
	"""
	
	#The nodes of the quadrature formula. 
	X = self.quadrature_object.nodes

	#Input to Gauss Laguerre quadrature formula.
	alpha = self.config.alpha
	integrand = ( 
	    (X/(2. * alpha) + 1)**q * (X**2/(4*alpha**2) + X/alpha)**abs(m) 
	    * self.get_laguerre(nu_prime - abs(m), 2 * abs(m))
	    * self.get_laguerre(nu - abs(m), 2 * abs(m)))

	integral = self.quadrature_object.integrate(integrand)

	#Normalization factor for <U|.
	N_bra = self.find_N(m, nu_prime)

	#Normalization factor for |U>.
	N_ket = self.find_N(m, nu)
	
	#All together now.
	d = N_bra * N_ket * integral / (2 * alpha) 
	
	return d

    def find_d_tilde(self, q, m, mu_prime, mu):
	"""
	d_tilde = find_d_tilde(q, m, mu_prime, mu)
	
	Evaluates the 'd_tilde' part of the matrix element, as described in eq. (A5),
	eq. (A11) and eq. (A12) in Kamta2005.

	Parameters
	----------
	q : int, 0 or 2, described in eq. (A5) in Kamta2005.
	m : int, quantum number, corresponding to the electron's 
	    angular momentum projection onto the z axis.
	mu_prime : int, 'eta' quantum number for the <bra| basis function.
	mu : int, 'eta' quantum number for the |ket> basis function.

	Returns
	-------
	d_tilde : complex, eta integral part of matrix element.
	"""
	def f_1(m, mu):
	    """
	    First square root in eq. (A11) in Kamta2005.
	    """
	    factor = sqrt((mu - m + 1.) * (mu + m + 1.) 
		/ (2. * mu + 1.) / (2. * mu + 3.))
	    return factor
	
	def f_2(m, mu):
	    """
	    Second square root in eq. (A11) in Kamta2005.
	    """
	    factor = sqrt((mu - m) * (mu + m) 
		/ (2. * mu - 1.) / (2. * mu + 1.))
	    return factor


	if q == 0:
	    #Simple integral of product of two V basis functions returns the 
	    #Kronecker delta, i.e. 1 if mu_prime equals mu, and zero if not.
	    d_tilde = float(mu_prime == mu) 
	    return d_tilde
	
	elif q == 1:

	    #Putting together the recurrence formula 
	    #to solve int(V' * V * eta).
	    d_tilde = (
		f_1(m, mu) * (mu_prime  == mu + 1) +
		f_2(m, mu) * (mu_prime  == mu - 1))

	    return d_tilde


	elif q == 2:

	    #Putting together the recurrence formula 
	    #to solve int(V' * V * eta**2).
	    d_tilde = (
		f_1(m, mu_prime) * f_1(m, mu) * (mu_prime + 1 == mu + 1) +
		f_1(m, mu_prime) * f_2(m, mu) * (mu_prime + 1 == mu - 1) +
		f_2(m, mu_prime) * f_1(m, mu) * (mu_prime - 1 == mu + 1) +
		f_2(m, mu_prime) * f_2(m, mu) * (mu_prime - 1 == mu - 1)) 

	    return d_tilde	
	
	elif q == 3:
	    #Putting together the recurrence formula 
	    #to solve int(V' * V * eta**3).
	    d_tilde = (
		f_1(m, mu_prime) * (
		    f_1(m, mu) * f_1(m, mu + 1) * (mu_prime + 1 == mu + 2) +
		    f_1(m, mu) * f_2(m, mu + 1) * (mu_prime + 1 == mu) +
		    f_2(m, mu) * f_1(m, mu - 1) * (mu_prime + 1 == mu) +
		    f_2(m, mu) * f_2(m, mu - 1) * (mu_prime + 1 == mu - 2)) +
		f_2(m, mu_prime) * (		
		    f_1(m, mu) * f_1(m, mu + 1) * (mu_prime - 1 == mu + 2) +
		    f_1(m, mu) * f_2(m, mu + 1) * (mu_prime - 1 == mu) +
		    f_2(m, mu) * f_1(m, mu - 1) * (mu_prime - 1 == mu) +
		    f_2(m, mu) * f_2(m, mu - 1) * (mu_prime - 1 == mu - 2)))

	    return d_tilde


	else:
	    raise NotImplementedError(
		"Only q <= 3 have been implemented.")

    def plot_xy(self, psi, r_max = 30, coordinates = "cartesian", 
	    display_plot = True, return_data = False):
	"""
    	(grid_1, grid_2, probability) = plot_xy(psi, r_max = 30,
	    coordinates = "cartesian", display_plot = True, 
	    return_data = False)

	Plots the probability distribution of <psi> in the xy plane. 
	May also return the visaulization data.

	Parameters
	----------
	psi : 1D complex array, containing the wavefunction that is to be 
	    visualized.
	r_max : float. Tells how far out one will plot. Default is 30 a.u.
	coordinates : ["carthesian" | "PSC"], chooses whether to plot in a
	    cartesian or a prolate spheroidal coordinate system. "cartesian" 
	    is default.
	display_plot : boolean, defaultly 'True', 
	    determines whether to show the figure.
	return_data : boolean, dafaultly 'False', 
	    determies whether the method should return the plotting data.

	Returns
	-------
	(OBS: only if <return_data> is set to 'True')
	grid_1 : 1D float array. x/xi grid, depending on coordinates.
	grid_2 : 1D float array. y/phi grid, depending on coordinates.
	probability : 2D float array. Probability distribution.
	"""
	#Defining appropriate grids.
	xi_max = sqrt(1 + 4 * r_max**2 / self.config.R**2)
	xi_grid = linspace(1, xi_max, 10 * xi_max)

	eta_grid = array([0])

	phi_grid = linspace(0, 2 * pi, 10 * r_max) 
	
	#Find probability inPSC.
	prob_psc = self.probability_on_PSC_grid(psi, 
		xi_grid, eta_grid, phi_grid)
	
	#Cartesian coordinates.
	if coordinates == "cartesian":
	    #Defining appropriate grids.
	    x_grid = linspace(-r_max, r_max, 20 * r_max)
	    y_grid = linspace(-r_max, r_max, 20 * r_max)
	    
	    #Initialize result array.
	    prob_cartesian = zeros([len(y_grid), len(x_grid)])
	    
	    #Interpolation function in PSC.
	    interp_prob = RectBivariateSpline(xi_grid, phi_grid, 
		    prob_psc[:,0,:], kx=1, ky=1, s = 0)

	    #Looping over x and y values.
	    for i, x in enumerate(x_grid):
		for j, y in enumerate(y_grid):
		    #Finding coresponding xi and phi values.
		    xi = self.find_xi(x, y, 0)
		    phi = self.find_phi(x, y, 0)

		    #Evaluate probability for said xi and phi.
		    prob_cartesian[j,i] = interp_prob(xi, phi) 
	    
	    if display_plot:
		#Plot xy plane probability in a new figure. Label axes.
		figure()
		pcolormesh(x_grid, y_grid, prob_cartesian)
		xlabel("x")
		ylabel("y")
		title("xy plane - $\eta$ = 0")
		show()
	    if return_data:
		#Returns grids and probability distribution.
		return x_grid, y_grid, prob_cartesian
	    	
	#Prolate Spheroidal Coordinates, PSC.
	elif coordinates == "PSC":
	    if display_plot:
		#Plot xy plane probability in a new figure. Label axes.
		figure()
		pcolormesh(phi_grid * 180 / pi, xi_grid, prob_psc[:,0,:])
		xlabel("$\phi$")
		ylabel("xi")
		title("xy plane - $\eta$ = 0")
		show()
	    if return_data:
		#Returns grids and probability distribution.
		return xi_grid, phi_grid, prob_psc[:,0,:]
	
	else:
	    raise IOError("This coordinate option is not valid.")


    def plot_xz(self, psi, r_max = 30, coordinates = "cartesian", 
	    display_plot = True, return_data = False):
	"""
    	(grid_1, grid_2, probability) = plot_xz(psi, r_max = 30,
	    coordinates = "cartesian", display_plot = True, 
	    return_data = False)

	Plots the probability distribution of <psi> in the xz plane. 
	May also return the visaulization data.

	Parameters
	----------
	psi : 1D complex array, containing the wavefunction that is to be 
	    visualized.
	r_max : float. Tells how far out one will plot. Default is 30 a.u.
	coordinates : ["carthesian" | "PSC"], chooses whether to plot in a
	    cartesian or a prolate spheroidal coordinate system. "cartesian" 
	    is default.
	display_plot : boolean, defaultly 'True', 
	    determines whether to show the figure.
	return_data : boolean, dafaultly 'False', 
	    determies whether the method should return the plotting data.

	Returns
	-------
	(OBS: only if <return_data> is set to 'True')
	grid_1 : 1D float array. x/xi grid, depending on coordinates.
	grid_2 : 1D float array. z/eta grid, depending on coordinates.
	probability : 2D float array. Probability distribution. 

	Notes
	-----
	When returning the data in the PSC system, 
	the two different phi values, 0 and pi,
	are made into a sign change in xi instead.
	"""
	#Defining appropriate grids.
	xi_max = 2 * r_max / self.config.R
	xi_grid = linspace(1, xi_max, 10 * xi_max)


	eta_grid = linspace(-1, 1, 10 * r_max)

	phi_grid = array([0, pi]) 
	
	#Find probability inPSC.
	prob_psc = self.probability_on_PSC_grid(psi, 
		xi_grid, eta_grid, phi_grid)
	
	#Cartesian coordinates.
	if coordinates == "cartesian":
	    #Defining appropriate grids.
	    x_grid = linspace(-r_max, r_max, 20 * r_max)
	    z_grid = linspace(-r_max, r_max, 20 * r_max)
	    
	    #Initialize result array.
	    prob_cartesian = zeros([len(z_grid), len(x_grid)])
	    
	    #Interpolation functions in PSC.
	    #phi = 0
	    interp_prob_0 = RectBivariateSpline(xi_grid, eta_grid, 
		    prob_psc[:,:,0], kx=1, ky=1, s = 0)
	    #phi = pi
	    interp_prob_pi = RectBivariateSpline(xi_grid, eta_grid, 
		    prob_psc[:,:,1], kx=1, ky=1, s = 0)

	    #Looping over x and z values.
	    for i, x in enumerate(x_grid):
		for j, z in enumerate(z_grid):
		    #Finding coresponding xi and eta values.
		    xi = self.find_xi(x, 0, z)
		    eta = self.find_eta(x, 0, z)

		    #Evaluate probability for said xi and eta.
		    #phi = 0.
		    if x > 0:
			prob_cartesian[j,i] = interp_prob_0(xi, eta) 
		    #phi = pi.
		    else:
			prob_cartesian[j,i] = interp_prob_pi(xi, eta) 
	    
	    if display_plot:
		#Plot xz plane probability in a new figure. Label axes.
		figure()
		pcolormesh(x_grid, z_grid, prob_cartesian)
		xlabel("x")
		ylabel("z")
		title("xz plane - $\phi$ = 0,$\pi$")
		show()
	    if return_data:
		#Returns grids and probability distribution.
		return x_grid, z_grid, prob_cartesian
	    	
	#Prolate Spheroidal Coordinates, PSC.
	elif coordinates == "PSC":
	    if display_plot:
		
		#Patcing phi = 0 & pi together.
		patched_xi_grid = r_[-1 * xi_grid[::-1], xi_grid]
		patched_prob = r_[flipud(prob_psc[:,:,1]), prob_psc[:,:,0]]
		
		#Plot xz plane probability in a new figure. Label axes.
		figure()
		pcolormesh(patched_xi_grid, eta_grid, transpose(patched_prob))

		xlabel("xi")
		ylabel("$\eta$")
		title("xz plane - $\phi$ = 0,$\pi$")
		show()
	    if return_data:
		#Returns grids and probability distribution.
		return patched_xi_grid, phi_grid, patched_prob
	
	else:
	    raise IOError("This coordinate option is not valid.")

    
    def plot_yz(self, psi, r_max = 30, coordinates = "cartesian", 
	    display_plot = True, return_data = False):
	"""
    	(grid_1, grid_2, probability) = plot_yz(psi, r_max = 30,
	    coordinates = "cartesian", display_plot = True, 
	    return_data = False)

	Plots the probability distribution of <psi> in the yz plane. 
	May also return the visaulization data.

	Parameters
	----------
	psi : 1D complex array, containing the wavefunction that is to be 
	    visualized.
	r_max : float. Tells how far out one will plot. Default is 30 a.u.
	coordinates : ["carthesian" | "PSC"], chooses whether to plot in a
	    cartesian or a prolate spheroidal coordinate system. "cartesian" 
	    is default.
	display_plot : boolean, defaultly 'True', 
	    determines whether to show the figure.
	return_data : boolean, dafaultly 'False', 
	    determies whether the method should return the plotting data.

	Returns
	-------
	(OBS: only if <return_data> is set to 'True')
	grid_1 : 1D float array. y/xi grid, depending on coordinates.
	grid_2 : 1D float array. z/eta grid, depending on coordinates.
	probability : 2D float array. Probability distribution. 

	Notes
	-----
	When returning the data in the PSC system, 
	the two different phi values, pi/2 and 3*pi/2,
	are made into a sign change in xi instead.
	"""
	#Defining appropriate grids.
	xi_max = 2 * r_max / self.config.R
	xi_grid = linspace(1, xi_max, 10 * xi_max)

	eta_grid = linspace(-1, 1, 10 * r_max)

	phi_grid = array([pi/2, 3*pi/2]) 
	
	#Find probability inPSC.
	prob_psc = self.probability_on_PSC_grid(psi, 
		xi_grid, eta_grid, phi_grid)
	
	#Cartesian coordinates.
	if coordinates == "cartesian":
	    #Defining appropriate grids.
	    y_grid = linspace(-r_max, r_max, 20 * r_max)
	    z_grid = linspace(-r_max, r_max, 20 * r_max)
	    
	    #Initialize result array.
	    prob_cartesian = zeros([len(z_grid), len(y_grid)])
	    
	    #Interpolation functions in PSC.
	    #phi = pi/2
	    interp_prob_0 = RectBivariateSpline(xi_grid, eta_grid, 
		    prob_psc[:,:,0], kx=1, ky=1, s = 0)
	    #phi = 3*pi/2
	    interp_prob_pi = RectBivariateSpline(xi_grid, eta_grid, 
		    prob_psc[:,:,1], kx=1, ky=1, s = 0)

	    #Looping over y and z values.
	    for i, y in enumerate(y_grid):
		for j, z in enumerate(z_grid):
		    #Finding coresponding xi and eta values.
		    xi = self.find_xi(0, y, z)
		    eta = self.find_eta(0, y, z)

		    #Evaluate probability for said xi and eta.
		    #phi = pi/2.
		    if y > 0:
			prob_cartesian[j,i] = interp_prob_0(xi, eta) 
		    #phi = 3*pi/2.
		    else:
			prob_cartesian[j,i] = interp_prob_pi(xi, eta) 
	    
	    if display_plot:
		#Plot yz plane probability in a new figure. Label axes.
		figure()
		pcolormesh(y_grid, z_grid, prob_cartesian)
		xlabel("y")
		ylabel("z")
		title("yz plane - $\phi$ = $\pi$/2,3$\pi$/2")
		show()
	    if return_data:
		#Returns grids and probability distribution.
		return y_grid, z_grid, prob_cartesian
	    	
	#Prolate Spheroidal Coordinates, PSC.
	elif coordinates == "PSC":
	    if display_plot:
		
		#Patcing phi = 0 & pi together.
		patched_xi_grid = r_[-1 * xi_grid[::-1], xi_grid]
		patched_prob = r_[flipud(prob_psc[:,:,1]), prob_psc[:,:,0]]
		
		#Plot yx plane probability in a new figure. Label axes.
		figure()
		pcolormesh(patched_xi_grid, eta_grid, transpose(patched_prob))
		xlabel("xi")
		ylabel("$\eta$")
		title("yz plane - $\phi$ =$\pi$/2, 3$\pi$/2")
		show()
	    if return_data:
		#Returns grids and probability distribution.
		return patched_xi_grid, phi_grid, patched_prob
	
	else:
	    raise IOError("This coordinate option is not valid.")
     
    
    def find_xi(self, x, y, z):
	"""
	xi = find_xi(x, y, z)

	Finds the xi value given by the cartesian point (x, y, z).

	Parameters
	----------
	x : float, the x value of a point in cartesian space.
	y : float, the y value of a point in cartesian space.
	z : float, the z value of a point in cartesian space.

	Returns
	-------
	xi : float, the xi value of the same point in PSC.
	"""
	R = self.config.R
	
	#Calculate xi value.
	#xi = (r1 + r2)/R.
	xi = (sqrt(x**2 + y**2 + (z - R/2.)**2) 
	    + sqrt(x**2 + y**2 + (z + R/2.)**2))/R
	
	return xi


    def find_eta(self, x, y, z):
	"""
	eta = find_eta(x, y, z)

	Finds the eta value given by the cartesian point (x, y, z).

	Parameters
	----------
	x : float, the x value of a point in cartesian space.
	y : float, the y value of a point in cartesian space.
	z : float, the z value of a point in cartesian space.

	Returns
	-------
	eta : float, the eta value of the same point in PSC.
	"""
    	R = self.config.R
	
	#Calculate eta value.
	#eta = (r1 - r2)/R.
	eta = (sqrt(x**2 + y**2 + (z - R/2.)**2) 
	    -  sqrt(x**2 + y**2 + (z + R/2.)**2))/R
	

	return eta


    def find_phi(self, x, y, z):
	"""
	phi = find_phi(x, y, z)

	Finds the phi value given by the cartesian point (x, y, z).

	Parameters
	----------
	x : float, the x value of a point in cartesian space.
	y : float, the y value of a point in cartesian space.
	z : float, the z value of a point in cartesian space.

	Returns
	-------
	phi : float, the phi value of the same point in PSC.
	"""
	#Calculates and returns the phi value.
	my_angle = arctan2(y,x)
	
	#Move from [-pi,pi] domain, to [0,2 * pi] domain.
	if my_angle < 0:
	    my_angle += 2 * pi

	return my_angle
    
    def tabulate_laguerre(self):
	"""
	tabulate_laguerre()

	Evaluates all relevant Laguerre polynomials in all the quadrature 
	points. Sets the table as a class/instance variable.

	Notes
	-----
	The first axis is the m axis. The index is order/2, 
	i.e. only even ms are included.
	The second axis is the nu axis. The index gives the degree.
	The third axis is the x axis. The polynomials are evaluated in the 
	nodes of the [2 + m_max + nu_max] order Gauss Laguerre 
	quadrature formula.
	"""
	#Shorthand for useful parameters. For convenience.
	m_max = self.config.m_max
	nu_max = self.config.nu_max
	X = self.quadrature_object.nodes
	
	#Initiate array.
	laguerre_table = zeros([
	    2 * m_max + 1, 
	    m_max + nu_max + 3, 
	    self.rule_order])

	for i, order in enumerate(range(0, 2 * m_max + 1, 2)):
	    for j, degree in enumerate(range(nu_max + m_max + 3)):
		#Alternative 1 #TODO 
		#Warning: This method is unstable for high degree/order.
#		laguerre_table[i,j,:] = genlaguerre(degree, order)(X)
		#-------------
		#Alternative 2
		#More stable, but sloooow, 
		#and (as of yet) only for m = 0.
		#TODO Could probably be more effective.
		if order == 0:
		    for k, x in enumerate(X):
			p1 = 1.0
			p2 = 0.0
			
			#Loop the recurrence relation to get the Laguerre polynomial 
			#evaluated at x.
			for l in range(1, degree + 1):
			    p3 = p2
			    p2 = p1
			    p1 = ((2 * l - 1 - x) * p2 - (l - 1) * p3)/l

			laguerre_table[i,j,k] = p1
		else:
		    laguerre_table[i,j,:] = genlaguerre(degree, order)(X)
		#---------------


	self.laguerre_table = laguerre_table

    def find_N(self, m, nu):
	"""
	N = find_N(m, nu)

	Numerically stable way of finding the normalization factor.

	Parameters
	----------
	m : integer, the angular momentum projection quantum number.
	nu : integer, some sort of quantum number associated with U.
	
	Returns
	-------
	N : float, normalization factor for the U basis function.

	Notes
	-----
	The fraction a!/b! is calculated using log and exp, in order
	to keep it all numerically stable.
	"""
	m = abs(m)

    	#Normalization factor for U.
	N = sqrt((2 * self.config.alpha)**(2 * m + 1) 
		* exp(self.lnfact(nu - m) - self.lnfact(nu + m)))

	return N
    

    def find_M(self, m, mu):
	"""
	M = find_M(m, mu)

	Numerically stable way of finding the normalization factor.

	Parameters
	----------
	m : integer, the angular momentum projection quantum number.
	mu : integer, some sort of quantum number associated with V.
	
	Returns
	-------
	M : float, normalization factor for the V basis function.

	Notes
	-----
	The fraction a!/b! is calculated using log and exp, in order
	to keep it all numerically stable.
	"""

    	#Normalization factor for V.
	M =  sqrt((0.5 + mu) * exp(self.lnfact(mu - m) - self.lnfact(mu + m)))

	return M


    def lnfact(self, a):
	    """
	    b = lnfact(a)

	    Calculates ln(a!).

	    Parameters
	    ----------
	    a : positive integer.

	    Returns
	    -------
	    b : float, ln(a!).
	    """
	    #Exploiting the rule ln(n*m) = ln(n) + ln(m).	    
	    b = sum(log(range(1, a + 1)))

	    return b
