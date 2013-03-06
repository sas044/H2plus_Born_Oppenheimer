"""
A number of different ways to distribute the B-splines in the basis.
Make your own! The only limit is your imagination. 
"""


from numpy import exp, sin, pi, arcsin


class Breakpoint_parameters:

    def __init__(self, rmin, rmax, n, distribution_type, 
	gamma=0.0, join_point=0):
	self.xmin = rmin
	self.xmax = rmax
	self.xsize = n
	self.distribution_type = distribution_type
	self.gamma = gamma 
	self.join_point = join_point


def create_breakpoint_sequence(params):
    """
    xi = create_breakpoint_sequence(params)

    Creates a sequence of breakpoints, using the details specified in 
    <params>.

    Parameters
    ----------
    params : Breakpoint_parameters instance, containing the relevant info 
	about the B-spline basis.

    Returns
    -------
    xi : 1D float array, containing the breakpoints.

    Notes
    -----
    See Bachau2001 for more on the B-spline basis and its breakpoints.
    """

    if(params.distribution_type == 'linear'):
	return linear_breakpoint_sequence(params.xmin,
	    params.xmax, params.xsize)

    elif(params.distribution_type == 'exponential'):
	return exponential_breakpoint_sequence(params.xmin, 
	    params.xmax, params.xsize, params.gamma)

    elif(params.distribution_type == 'quadraticlinear'):
	return quadraticlinear_breakpoint_sequence(params.xmin, 
	    params.xmax, params.xsize, params.joinpoint)

    elif(params.distribution_type == 'sinelike'):
	return sinelike_breakpoint_sequence(params.xmin, 
	    params.xmax, params.xsize, params.alpha)

    elif(params.distribution_type == 'arcsine'):
	return arcsine_breakpoint_sequence(params.xmin, 
	    params.xmax, params.xsize)
    else:
	raise NameError, "Sequence not recognized!"
	


def linear_breakpoint_sequence(rmin, rmax, n):
    """
    xi = linear_breakpoint_sequence(rmin, rmax, n)

    Compute linear breakpoint sequence on the interval 
    [rmin, rmax] with n points.

    Parameters
    ----------
    rmin : float, lowest value in the domain.
    rmax : float, largest value in the domain.

    Returns
    -------
    xi : float list, containing the breakpoint sequence of the B-spline basis.
    """
    #Create sequence.
    xi = [rmin + (rmax - rmin) * i / float(n - 1) for i in range(n)]

    return xi



def exponential_breakpoint_sequence(rmin, rmax, n, gamma):
    """
    Compute exponential breakpoint sequence on the interval [rmin, rmax] 
    with n points. The parameter gamma determines point spacing:

	gamma -> 0    =>  linear sequence
	gamma -> \inf =>  all points exponentially close to rmin

    Breakpoints are computed according to

				       exp(gamma*(i-1)/(n-1)) - 1
	xi_i = rmin + (rmax - rmin) * ----------------------------
					    exp(gamma) - 1
    """

    h = rmax - rmin
    #Create sequence.
    xi = [rmin + h * (exp(gamma * (i - 1.0)/(n - 1.0)) - 1)/(exp(gamma) - 1)
	for i in range(1, n + 1) ]

    return xi


def quadraticlinear_breakpoint_sequence(rmin, rmax, n, join_point):
    """
    Compute quadratic/linear breakpoint sequence on the interval 
    [rmin, rmax] with n points. The sequence consists of quadratic 
    and linear sub-sequences, joined at	'join_point'. 
    The quadratic sequence extends from 'rmin' to 'join_point'-1, 
    while the linear sequence extends from 'join_point' to 'rmax'. 
    This sequence is particularly suited for problems where both 
    bound and continuum states needs to be resolved.
    """
    
    #join_point = join_point + 1
    join_point = float(join_point)
    rmin = float(rmin)
    rmax = float(rmax)

    #Compute first point of sequence
    r0 = (rmax * (join_point - 1) + rmin * (n - join_point)) / (
	2.0 * n - join_point - 1.0)

    #Scaling parameters for the two parts
    alpha = (r0 - rmin) / float(join_point - 1)**2
    beta = (rmax - r0) / float(n - join_point)
    
    #Parabolic part of sequence
    xi = [rmin + alpha * (i - 1)**2 for i in range(1, join_point)]
    
    #Linear part of sequence
    xi += [r0 + beta * (i - join_point) for i in range(join_point, n+1)]


    return xi


def sinelike_breakpoint_sequence(rmin, rmax, n, alpha):
    """
    Set up a sineline breakpoint sequence on the interval [rmin, rmax] with 
    n points. The parameter 'alpha' determines whether grid points cluster:

	alpha -> 0     =>   cluster at rmax
        alpha -> inf   =>   cluster at rmin

    Intermediate values, such as alpha ~ 2, results in clustering of points 
    and both ends.

    Formula:

	      xi_i = r_min + r_rmax * sin(pi/2 * ((i - 1)/(n - 1))**alpha )
    """

    # Create sequence
    xi = [rmin + rmax * sin(pi / 2.0 * ((i - 1.0) / (n - 1.0))**alpha
	) for i in range(1, n + 1)]

    return xi


def arcsine_breakpoint_sequence(rmin, rmax, n):
    """
    Set up a arcsineline breakpoint sequence on the interval [rmin, rmax] with 
    n points.
    """

    #Create sequence
    xi = [rmin + (rmax - rmin) / pi * arcsin(1.0 / i) for i in range(n)]

    return xi

