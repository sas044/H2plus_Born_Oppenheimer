import breakpointsequences as breakpoints
from numpy import linalg, diag, array, zeros, shape, sqrt, ones, sort
import numpy
import tables

class Bspline:
    """
    This class contains the basic methods of a B-spline basis, 
    such as setting up the basis, evaluating splines and derivatives of 
    splines and transforming a function from the B-spline basis to a 
    given grid. A more specialized method, that creates a table of spline 
    values to facilitatie quick Gauss Legendre integration, is also included.

    The work is based on the book
	C. de Boor, 'A Practical Guide to Splines', 
	Springer-Verlag, New York, 2001
    and the article
	H. Bachau et al. 'Applications of B-splines in atomic and molecular 
	physics.', Rep. Prog. Phys., 64:1815-1943, 2001.
    """

    def __init__(self, xmin, xmax, xsize, order, 
	continuity = "zero", distribution_type = "linear"):
	"""
	Bspline(xmin, xmax, xsize, order, 
	    continuity = "zero", distribution_type = "linear")
	
	Constructor. Sets up the Bspline instance.

	Parameters
	----------
	xmin : float, left end of the domain.
	xmax : float, right end of the domain.
	xsize : integer, approximate number of B-splines in the basis.
	order : integer, order of the spline functions.
	continuity : string, ["zero" (by default), "vanilla"]. Determines if
	    the expanded function can be != 0 at the endpoints.
	distribution_type : string, ["linear", (see breakpointsequences.py)],
	    description of how the splines are distributed on the domain.
	"""
	
	self.spline_order = order
	self.continuity_type = continuity
	self.number_of_breakpoints = xsize
	self.distribution_type = distribution_type

	breakpoint_parameters = breakpoints.Breakpoint_parameters(
	     xmin, xmax, xsize, distribution_type)
	
	self.breakpoint_sequence = breakpoints.create_breakpoint_sequence(
	    breakpoint_parameters)
	
	self.create_continuity_sequence()
	
	self.create_knot_sequence()

	self.number_of_bsplines = sum(self.spline_order 
	    - array(self.continuity_sequence)) - self.spline_order

    def evaluate_bspline(self, x, left_knot_point):
	"""
	spline_value = evaluate_bspline(x, left_knot_point)

	Evaluate B-spline starting at knot point 'left_knot_point' 
	in grid point x.

	Parameters
	----------
	x : float, where the B-spline should be evaluated.
	left_knot_point : integer, index of the spline's starting point in the
	    knot sequence.

	Returns
	-------
	spline_value : float, the value of the B-spline in x.
	"""
	return self.__evaluate_bspline__(x, self.spline_order, 
	    left_knot_point)


    def evaluate_bspline_derivative_1(self, x, left_knot_point):
	"""
	spline_value = evaluate_bspline_derivative_1(x, left_knot_point)

	Evaluate first derivate of B-spline using recursive formula. 

	Parameters
	----------
	x : float, where d/dx(B-spline) should be evaluated.
	left_knot_point : integer, index of the spline's starting point in the
	    knot sequence.

	Returns
	-------
	spline_value : float, the value of the differentiated B-spline in x.
	"""

	#Some shorthands
	k = self.spline_order
	knots = self.knot_sequence

	#Used to check for zero values
	eps = 1e-15

	#We need two different splines
	b_1 = self.__evaluate_bspline__(x, k-1, left_knot_point)
	b_2 = self.__evaluate_bspline__(x, k-1, left_knot_point + 1)

	if b_1 > eps:
	    a_ = (k - 1.0) / float(knots[left_knot_point + k - 1] 
		- knots[left_knot_point])
	    b_1 *= a_

	if b_2 > eps:
	    b_ = (k - 1.0) / float(knots[left_knot_point + k] 
		- knots[left_knot_point + 1])
	    b_2 *= b_

	return b_1 - b_2


    def evaluate_bspline_derivative_2(self, x, left_knot_point):
	"""
	spline_value = evaluate_bspline_derivative_2(x, left_knot_point)

	Evaluate second derivate of B-spline using recursive formula.

	Parameters
	----------
	x : float, where d2/dx2(B-spline) should be evaluated.
	left_knot_point : integer, index of the spline's starting point in the
	    knot sequence.

	Returns
	-------
	spline_value : float, the value of twice the differentiated 
	    B-spline in x.

	"""

	#Some shorthands.
	k = self.spline_order
	knots = self.knot_sequence

	#Used to check for zero values.
	eps = 1e-15

	#We need three different splines.
	b_1 = self.__evaluate_bspline__(x, k-2, left_knot_point)
	b_2 = self.__evaluate_bspline__(x, k-2, left_knot_point + 1)
	b_3 = self.__evaluate_bspline__(x, k-2, left_knot_point + 2)

	#Calculate derivate. Checking that return value is not zero, if so
	#avoid computing fractions, as they may contain zero denominators.
	bspline = 0.0
	
	if b_1 > eps:
	    b_ = (k - 2.0) / ( knots[left_knot_point + k - 2] 
		- knots[left_knot_point] )
	    b_1 *= b_

	if b_2 > eps:
	    c_ = (k - 2.0) / ( knots[left_knot_point + k - 1] 
		- knots[left_knot_point + 1] )
	    b_2 *= c_
	
	if b_3 > eps:
	    e_ = (k - 2.0) / ( knots[left_knot_point + k] 
		- knots[left_knot_point + 2] )
	    b_3 *= e_

	if (b_1 > eps) | (b_2 > eps):
	    a_ = (k - 1.0) / ( knots[left_knot_point + k - 1] 
		- knots[left_knot_point] )
	    bspline += a_ * (b_1 - b_2)

	if (b_2 > eps) | (b_3 > eps):
	    d_ = (k - 1.0) / ( knots[left_knot_point + k] 
		- knots[left_knot_point + 1] )
	    bspline -= d_ * (b_2 - b_3)

	return bspline


    def __evaluate_bspline__(self, x, spline_order, left_knot_point):
	"""
	spline_value = evaluate_bspline(x, left_knot_point)

	Evaluate B-spline value B(x) at point 'x' over knot sequence. B-spline
	is of order 'spline_order'. B(x) is non-zero over the interval defined
	by 'left_knot_point' and 'left_knot_point' + 'spline_order'.

	The calculation is performed using the three-point recursion formula
	for splines,
			
	B(k, i, x) = (x - t_i) / (t_(i+k-1) - t_i) * B(k-1, i, x)
	    + (t_(i+k) - x) / (t_(i+k) - t(i+1)) * B(k-1, i+1, x)

	where it is understood that 

		 B(1, i, x) = 1  , t_i <= x < t_(i+1)
		 B(1, i, x) = 0  , otherwise

	Parameters
	----------
	x : float, where the B-spline should be evaluated.
	left_knot_point : integer, index of the spline's starting point in the
	    knot sequence.

	Returns
	-------
	spline_value : float, the value of the B-spline in x.
	"""

	#Number smaller than this is treated as 0
	eps = 1e-15


	#If required spline is of order 1, check requirement on knot 
	#points and x, returning B = 1 if all is good, otherwise 0.
	if spline_order == 1:
	    
	    if self.knot_sequence[left_knot_point] <= x < self.knot_sequence[
		left_knot_point + 1]:
		    b_1_0 = 1.0
	    else:
		    b_1_0 = 0.0

	#Compute b-spline of order > 1. We must check that b-spline values 
	#returned from next recursion call is not zero, in which case we 
	#avoid 0/0-problem in recursion formula by removing offending term.
	else:

	    #Shorthands for the actual knot sequence values needed here
	    start_knot_0 = self.knot_sequence[left_knot_point]
	    end_knot_0 = self.knot_sequence[left_knot_point + spline_order-1]
	    start_knot_1 = self.knot_sequence[left_knot_point + 1]
	    end_knot_1 = self.knot_sequence[left_knot_point + spline_order]

	    #Recurse
	    b_0_0 = self.__evaluate_bspline__(x, 
		spline_order - 1, left_knot_point)
	    b_0_1 = self.__evaluate_bspline__(x, 
		spline_order - 1, left_knot_point + 1)
	    
	    #If b_1_0 or b_0_0 is zero, endKnot - startKnot will be zero also.
	    #Check this to avoid division by zero.
	    b_1_0 = 0.0
	    if b_0_0 > eps:
		b_1_0 += (x - start_knot_0) / float(end_knot_0 
		    - start_knot_0) * b_0_0
	    if b_0_1 > eps:
		b_1_0 += (end_knot_1 - x) / float(end_knot_1 
		    - start_knot_1) * b_0_1

	return b_1_0


    def evaluate_bspline_derivative_on_grid(self, spline_start_point, 
	grid, order):
	"""
	spline_values = evaluate_bspline_derivative_on_grid(
	    spline_start_point, grid, order)

	Evaluates derivate of a B-spline on given grid.

	Parameters
	----------
	spline_start_point : integer, index of the spline's starting point 
	    in the knot sequence.
	grid : 1D float array, the x values in which the differentiated 
	    B-spline should be evaluated.
	order : integer, the number of differentiations.

	Returns
	-------
	spline_values : 1D float array, the values of the B-spline on <grid>.
	"""

	#Creates the <order> times differentiation method.
	dnb = eval('self.evaluate_bspline_derivative_%i' % order)
	
	#Initializes return array.
	f = zeros(len(grid))

	#Loops over grid.
	for p, x in enumerate(grid):
	    f[p] = dnb(x, spline_start_point)

	return f


    def evaluate_bspline_on_grid(self, spline_start_point, **args):
	"""
	grid, b_spline = evaluate_bspline_on_grid(spline_start_point, **args)

	Evaluate a B-spline on given grid.

	Parameters
	----------
	spline_start_point : integer, index of the spline's starting point 
	    in the knot sequence.
	*grid* : 1D float array, the x values in which the B-spline should 
	    be evaluated.
	*number_of_grid_points* : integer, the number of points wanted on 
	    the grid.
	
	Returns
	-------
	grid : 1D float array, the x values in which the B-spline were
	    evaluated.
	b_spline : 1D float list, the values of the B-spline on <grid>.
	"""
	#Initializes return list.
	b_spline = []
	
	#Reads **arguments, and create grid.
	if 'grid' in args:
	    grid = args['grid']
	elif 'number_of_grid_points' in args:
	    x_min = self.breakpoint_sequence[0]
	    x_max = self.breakpoint_sequence[-1]
	    number_of_grid_points = args["number_of_grid_points"]
	    grid = numpy.linspace(x_min, x_max, number_of_grid_points)
	
	#Looping over the grid.
	for x in grid:
	    # Create B-spline at x.
	    b_spline.append(self.__evaluate_bspline__(
		x, self.spline_order, spline_start_point))
	
	return grid, b_spline


    def create_continuity_sequence(self):
	"""
	create_continuity_sequence()

	Create a sequence [v_i] defining the continuity condition at each 
	breakpoint. The continuity condition is C^(v_i-1), where v_i is 
	associated with	breakpoint i. 
	C^k-continuity implies continuous k'th derivative. 
	C^-1 means that the function (spline) itself is discontinuous.
	"""

	n_ = self.number_of_breakpoints

	# Vanilla: maximum continuity at interior points,
	#          maximum multiplicity at end points.
	if(self.continuity_type == 'vanilla'):
	    self.continuity_sequence = (self.spline_order - 1) * ones(n_, 
		dtype='int')
	    self.continuity_sequence[0] = 0
	    self.continuity_sequence[-1] = 0

	# Zero: maximum continuity at interior points,
	#       maximum multiplicity - 1 at end points.
	#       This will remove the nonzero spline and
	#       endpoints, giving zero boundary conditions.
	if(self.continuity_type == 'zero'):
	    self.continuity_sequence = (self.spline_order - 1) * ones(n_, 
		dtype='int')
	    self.continuity_sequence[0] = 1
	    self.continuity_sequence[-1] = 1

    def create_knot_sequence(self):
	"""
	create_knot_sequence()

	Create a B-spline knot sequence given a breakpoint sequence, 
	continuity conditions at each point (continuitySequence) and the 
	order of the B-splines.
	"""
	
	self.knot_sequence =  []
	n_ = self.number_of_breakpoints

	for i in range(n_):
	    knot_point_multiplicity = (self.spline_order 
		- self.continuity_sequence[i])
	    for j in range(knot_point_multiplicity):
		self.knot_sequence.append(self.breakpoint_sequence[i])

    

    def construct_function_from_bspline_expansion(self, c, grid):
	"""
	f = construct_function_from_bspline_expansion(c, grid)

	Given the sequence c of expansion coefficient in the B-spline basis,
	reconstruct corresponding function grid. The grid should not be of
	greater extent than the breakpoints.
	
	Parameters
	----------
	c : 1D float array, the B-spline representation of the function.
	grid : 1D float array, the x values in which the function should 
	    be evaluated.
	
	Returns
	-------
	f : 1D float array, the function on the grid.
	"""
	
	f = zeros(len(grid))
	for i, x in enumerate(grid):
	    for j, coeff in enumerate(c):
		f[i] += coeff * self.evaluate_bspline(x, j)

	return f


	    
    def make_bspline_table(self):
	"""
	make_bspline_table()

	Makes a table containing the nonzero values of all the splines 
	(and its derivative) that are needed to do the integrations.
	Also returns the x values the spline is evaluated in.

	Returns
	-------
	spline_table : 4D float array, containing the tabulated spline values.
	    The first axis spans the different splines, the second axis spans 
	    the spline intervals, the third axis spans the integration points,
	    and the fourth axis spans the zeroth and first derivative of the 
	    splines.
	x_values : 3D float array, containing the x values corresponding to 
	    the spline table.
	"""
	

	#Initializing table.
	spline_table = zeros([self.number_of_bsplines,
	    self.spline_order, self.spline_order,2])

	x_values = zeros(shape(spline_table)[:-1])
	
	
	#Finding the node distribution on the interval [-1,1].
	beta = [1.0 / (2.0 * sqrt(1.0 - (2.0 * n)**-2)) 
	    for n in range(1,self.spline_order)] 
	jacobi_matrix = diag(array(beta), -1) + diag(array(beta), 1)
	E, V = linalg.eig(jacobi_matrix) #Remove V?
	nodes = sort(E)
	
	#Number smaller than this is treated as 0.
	eps = 1e-15
	
	#Looping splines.
	for i in range(self.number_of_bsplines):
	    #Looping non-zero intervals.
	    for j in range(self.spline_order):
		#Looping integration points.
		for k in range(self.spline_order):
		    a = self.knot_sequence[i+j]
		    b = self.knot_sequence[i+j+1]
		    
		    if b - a < eps :
			continue
		    
		    spline_table[i,j,k,0] = self.evaluate_bspline(
			(nodes[k]*(b-a)/2+((a+b)/2)), i)
		    
		    x_values[i,j,k] = nodes[k]*(b-a)/2+((a+b)/2)
		    
		    spline_table[i,j,k,1] = \
			self.evaluate_bspline_derivative_1(
			(nodes[k]*(b-a)/2+((a+b)/2)), i)
			    
	
	return spline_table, x_values
		
    def save_spline_info(self, filename):
	"""
	save_spline_info(filename)

	Saves the info needed to recreate this instance of the spline class.

	Parameters
	----------
	filename : string, the name of the (existing) HDF5 file where the info
	    should be stored.

	Notes
	-----
	Any changes to this class must be reflected in load_spline_object().
	"""
	
	spline_info = [
	    self.breakpoint_sequence[0],
	    self.breakpoint_sequence[-1],
	    self.number_of_breakpoints,
	    self.spline_order,
	    self.continuity_type,
	    self.distribution_type
	    ]

	f = tables.openFile(filename,'r+')
	try:
	    f.createArray("/", "splineInfo", spline_info)
	finally:
	    f.close()

def load_spline_object(filename):
    """	
    load_spline_object(filename)

    Recreates a Bspline instance from data in <filename>. 

    Parameters
    ----------
    filename : string, the name of the HDF5 file where the info	is stored.

    Notes
    -----
    Any changes to this class must be reflected in save_spline_info().

    """
    
    #Retrieve spline info from file.
    f = tables.openFile(filename)
    try:
	spline_info = f.root.splineInfo[:]
    finally:
	f.close()
    
    #Initialize the spline object.
    spline_object = Bspline(
	    float(spline_info[0]),
	    float(spline_info[1]),
	    int(spline_info[2]),
	    int(spline_info[3]),
	    continuity = spline_info[4],
	    distribution_type = spline_info[5])
    
    return spline_object
				

