from numpy import r_, diag, argsort, linalg, zeros
from types import FunctionType


class Gauss_laguerre_quadrature_rule:
    """
    A simple class to handle 1-dimensional Gauss-Legendre integration.
    Integration nodes and weights for the inteval [0, inf] is computed
    based on given quadrature order via the tri-diagonal Jacobi matrix.

    The main function of this class is
    
	integrate(f),
	    
    which integrates a function f on the interval [a,b].
    """

    def __init__(self, rule_order):
	"""
	Gauss_laguerre_quadrature_rule(rule_order)

	Constructor. 

	Parameters
	----------
	rule_order : int, the order of the quadrature rule.
	    The intergration is supposed to be exact for polynomials up to 
	    degree 2 * <rule_order> - 1.
	"""
	self.rule_order = rule_order
	self.nodes = 0
	self.weights = 0
	
	#Alternative 1 --- TODO Unstable for higher orders. 
#	self.setup_jacobi_matrix()
#	self.compute_nodes_and_weights()
	#-----------------

	#Alternative 2 --- Better!
	self.alternative_node_and_weight_computation()
	#-----------------
    
    def integrate(self, f):
	"""
	result = integrate(f)

	Integrate function f on interval [0, inf] 
	using Gauss-Laguerre quadrature.

	Parameters
	----------
	f : a function, evaluates the integrand for arbitrary x points.

	Returns
	-------
	result : float, the integral of f * exp(-x) from 0 to infinity.
	"""

	#Perform gauss integration on [0, inf]
	I = 0.0

	if type(f) == FunctionType:
	    for i in range(self.rule_order):
		w = self.weights[i]
		x = self.nodes[i]
		I += w * f(x)
	elif len(f) == len(self.nodes):
	    I = sum(self.weights * f)

	else:
	    error_message = ("Input should be either a function or an " + 
		"array/list with the same length as self.nodes.")
	    raise IOError(error_message)
	
	return  I
	
	    

    def setup_jacobi_matrix(self):
	"""
	setup_jacobi_matrix()

	Setup the tri-diagonal Jacobi matrix corresponding to
	Leguerre polynomials.
	"""
    	
	i = r_[1 : self.rule_order + 1]
	#Diagonal.
	a = 2 * i - 1
	#Sub/super diagonals.
	b = i[:-1]


	self.jacobi_matrix = diag(b, -1) + diag(a) + diag(b, 1)
    

    def compute_nodes_and_weights(self):
	"""
	compute_nodes_and_weights()

	Obtain quadrature grid and weights from the eigenvectors
	and eigenvalues of the Jacobi matrix.

	Notes
	-----
	Seems to become unstable at approx. m = 16.
	"""
	
	#Find eigenvalues and eigenvectors.
	E, V = linalg.eig(self.jacobi_matrix)
	
	#Sort according to rising nodes.
	index_of_nodes = argsort(E)
	self.nodes = E[index_of_nodes]

	#The weights are the square of the first elements in the eigenvectors.
	self.weights = V[0,:]**2
	self.weights = self.weights[index_of_nodes]

			
    def alternative_node_and_weight_computation(self):
	"""
	Computes nodes and weights using Newton's method.

	Borrowing an algorithm from 
	"Numerical recipes in C: the art of scientific computing",
	(ISBN 0-521-43108-5), pp. 152-153, 
	alf = 0.
	"""
	#Maximal number of iterations of the Newton method.
	MAXIT = 10

	#Tolerance of the Newton method.
	EPS = 3e-14
	
	#Initializing arrays.
	x = zeros(self.rule_order + 1)
	w = zeros(self.rule_order + 1)
	
	#For convenience.
	n = self.rule_order
	
	#Looping over zeros of the Laguerre polynomial.
	for i in range(1, n+1):
	    if i == 1:
		#Approximation to the 1st root.
		z = 3.0/(1.0 + 2.4 * n)
	    elif i == 2:
		#Approximation to the 2nd root.
		z += 15.0/(1.0 + 2.5 * n)
	    else:
		#Approximation to the nth root.
		ai = i - 2
		z += (1.0 + 2.55 * ai)/(1.9 * ai) * (z - x[i-2])
	    
	    #Refinement by Newton's method.
	    for its in range(1,MAXIT+1):
		p1 = 1.0
		p2 = 0.0
		#Loop the recurrence relation to get the Laguerre polynomial 
		#evaluated at z.
		for j in range(1,n+1):
		    p3 = p2
		    p2 = p1
		    p1 = ((2 * j - 1 - z) * p2 - (j - 1) * p3)/j
		
		#p1 is now the desired Laguerre polynomial. 
		#We next compute pp, its derivative, by a standard relation 
		#involving also p2, the polynomial of one lower order.
		pp = n * (p1 - p2) / z
		z1 = z
		#Newton's formula.
		z = z1 - p1/pp
		if abs(z - z1) <= EPS:
		    break
	    
	    if its > MAXIT - 1:
		raise StandardError("Exceded maximal number of iterations")
	    #Store the root and the weight.
	    x[i] = z
	    w[i] = -1./(pp * n * p2)
	
	#Returning to sensible indexing.
	self.nodes = x[1:] 
	self.weights = w[1:] 



