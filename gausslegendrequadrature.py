from numpy import product, shape, isnan, sqrt, diag 
from numpy import array, argsort, linalg, exp, sin

import scipy.integrate

class Gauss_legendre_quadrature_rule:
    """
    A simple class to handle 1-dimensional Gauss-Legendre integration.
    Integration nodes and weights for the inteval [-1,1] are computed
    based on given quadrature order via the tri-diagonal Jacobi matrix.

    The main functions of this class are
    
	integrate(f, a, b)
	    and
	integrate_from_table(B1, B2, fox, x)
	    
    which integrate a function on the interval [a,b].
    """

    def __init__(self, order):
	"""
	Gauss_legendre_quadrature_rule(order)

	Constructor.

	Parameters
	----------
	order : integer, the order of the integration rule.
	"""
	self.rule_order = order
	self.nodes = 0
	self.weights = 0

	self.setup_jacobi_matrix()
	self.compute_nodes_and_weights()


    def integrate(self, f, a, b):
	"""
	I = integrate(f, a, b)
	    
	Integrate function f on interval [a,b] using Gauss-Legendre quadrature.

	Parameters
	----------
	f : function to be integrated.
	a : left end of domain.
	b : right end of domain.

	Returns
	-------
	I : float, the integral.
	"""

	#Perform gauss integration on [-1,1]
	I = 0.0
	for i in range(self.rule_order):
	    w = self.weights[i]
	    x = (self.nodes[i] - 1.0) * float(b - a) / 2.0 + b
	    I += w * f(x)
	
	#Scale to correct interval
	return (b - a) / 2.0 * I
	
	
	    
    def integrate_from_table(self, B1, B2, x, fox = None):
	"""
	I = integrate_from_table(B1, B2, x, fox)

	Calculate <B1|fox|B2>, using Gauss-Legendre quadrature, and tabulated 
	spline values.	
	
	Parameters
	----------
	B1 : 2D float array, containing the tabulated values of a spline.
	B2 : 2D float array, containing the tabulated values of a spline.
	x : 2D float array, x values organized the same way.
	fox : 2D float array, function values organized the same way.
	
	Returns
	-------
	I : float, the integral.
	"""
	
	#Perform gauss integration
	I = 0.0

	for i in range(product(shape(B1))/self.rule_order):
	    # (b-a), not very nice, I'm afraid...
	    interval_width = (x[i,-1] - x[i,0])*2.0/(self.nodes[-1] 
		- self.nodes[0])

	    #Handeling no function.
	    if fox == None:
		integrand = B1[i,:] * B2[i,:] 
	    else:
		integrand = fox[i,:] * B1[i,:] * B2[i,:]
	        
	    integral_part = sum(self.weights.ravel() * integrand.ravel()
		) * interval_width/ 2.0 
	    
	    if isnan(integral_part):
		    integral_part = 0
	    
	    I += integral_part 
    
	else:
	    return I


    def setup_jacobi_matrix(self):
	"""
	setup_jacobi_matrix()

	Setup the tri-diagonal Jacobi matrix corresponding to
	Legendre polynomials.
	"""

	beta = [1.0 / (2.0 * sqrt(1.0 - (2.0 * n)**-2)) 
	    for n in range(1, self.rule_order)] 
	self.jacobi_matrix = diag(array(beta), -1) + diag(array(beta), 1)


    def compute_nodes_and_weights(self):
	"""
	compute_nodes_and_weights()

	Obtain quadrature grid and weights from eigenvectors
	and eigenvalues of Jacobi matrix.
	"""

	E, V = linalg.eig(self.jacobi_matrix)

	self.nodes = (E)
	self.weights = 2.0 * V[0,:]**2
	
	index_of_nodes = argsort(self.nodes)
	
	self.nodes = self.nodes[index_of_nodes]
	self.weights = self.weights[index_of_nodes]
	

    def test_quad_rule(self):
	"""
	test_quad_rule()

	A test of computed nodes and weights. Integrates a few simple
	functions and compares with "exact" result.
	"""

	func = ['x', 'x**2', 'exp(x)', 'sin(x)']
	A = [-1.0, 1.0, 0.5, 1.3]
	B = [1.0, 2.5, 1.0, 2.0]
	
	I1 = []
	I2 = []

	print "f(x)\ta\t\tb\t\tI_gl\t\tI_scipy\t|Diff|"
	for i in range(len(func)):
	    f = lambda x : eval(func[i]) 
	    a = A[i]
	    b = B[i]
	    I1 = self.integrate(f, a, b)
	    I2 = scipy.integrate.quad(f, a, b)[0]

	    print "%s\t%f\t%f\t%f\t%f\t%f" % (
		func[i], a, b, I1, I2, abs(I1-I2)) 
