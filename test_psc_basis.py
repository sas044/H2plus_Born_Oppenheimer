"""
Testing battery to assure that the program 'psc_basis.py' is working properly.
"""
from numpy import *
from pylab import *
from math import factorial

import psc_basis
import gausslaguerrequadrature as quad
import time
from scipy.special import genlaguerre


def test_setup_overlap():
    """
    Tests if the overlap matrix is ok. Elements tend to get very large, especially 
    for complex scaling case.
    """
    t_1 = time.time()
    a = psc_basis.PSC_basis(m = 0, nu = 14, mu = 15, R = 2.0, beta = 1, theta= 0)
    #a = psc_basis.PSC_basis(m = 30, nu = 70, mu = 30, R = 2.0, beta = 0.2, theta= 0.1)
    
    a.setup_overlap()

    t_2 = time.time()
    S = a.overlap_matrix

    print "Imaginary part of the matrix:", sum(imag(S.ravel()))
    print "Symmetric? (should be zero):", sum(ravel(S - transpose(S)))
    print "Largest entry: ", max(S.ravel())
    print "Number of non-zero entries / total:", sum(abs(S) > 0), "/", S.shape[0]**2
    print "Time (min):", (t_2 - t_1)/60 

    figure()
    pcolormesh(abs(real(S))> 0)
    show()
    figure()
#    pcolormesh(log(abs(real(S + 1e-5))))
    pcolormesh(real(S))
    colorbar()
    show()

    return a

def test_plot_basis_functions():
    """
    At once a test of visualization success 
    and how the basis functions actually look.
    """
    #Pretty arbitrary parameters.
    a = psc_basis.PSC_basis(m = 3, nu = 3, mu = 3, R = 2.0, beta = 1, theta= 0)

    #Initialize a wavefunction array.
    psi = zeros(a.basis_size)

    #Which basis function to look at (the square of).
    psi[108] = 1

    #Prints  index | quantum numbers. 
    for i ,A in enumerate(a.index_iterator):
	print i, "|", A
	
    #Plots probability distribution in the xy, xz and yz planes,
    #in PSC and cartesian coordinates.
    a.plot_xy(psi,r_max = 30,coordinates = "PSC",return_data = True)
    colorbar()
    a.plot_xy(psi,r_max = 30,coordinates = "cartesian",return_data = True)
    colorbar()
    a.plot_xz(psi,r_max = 30,coordinates = "PSC",return_data = True)
    colorbar()
    a.plot_xz(psi,r_max = 30,coordinates = "cartesian",return_data = True)
    colorbar()
    a.plot_yz(psi,r_max = 30,coordinates = "PSC",return_data = True)
    colorbar()
    a.plot_yz(psi,r_max = 30,coordinates = "cartesian",return_data = True)
    colorbar()

def test_gauss(rule_order):
    """
    Tests to find out why the rule is bad.
    """

    my_rule = quad.Gauss_laguerre_quadrature_rule(rule_order)
    
    my_rule.setup_jacobi_matrix()
    my_rule.compute_nodes_and_weights()
    X = my_rule.nodes
    W = my_rule.weights

    my_rule.alternative_node_and_weight_computation()
    X_2 = my_rule.nodes
    W_2 = my_rule.weights

    return X, W, X_2, W_2
    
#    print "sum(x) = n^2:"
#    print sum(X), "\t",rule_order**2, "\t", sum(X) - rule_order**2, "\t", (sum(X) - rule_order**2)/ rule_order**2
#    print "prod(x) = n!:"
#    print prod(X), "\t", factorial(rule_order), "\t", prod(X) - factorial(rule_order), "\t", (prod(X) - factorial(rule_order))/factorial(rule_order)
#    return my_rule
#    

#
#    L = genlaguerre(rule_order,0)
#    L_1 = genlaguerre(rule_order + 1,0)
#    
#    X_corrected = X * 0
#    X_padded = 0.5 * (r_[0,X] + r_[X,X[-1] + 1])
#    
#
#    for i in range(len(X)):
#	x_left = X_padded[i]
#	x_right = X_padded[i + 1]
#	X_corrected[i] = bisection(L, x_left, x_right)
#
#    return my_rule, X, X_corrected
#
def bisection(L, x_left, x_right):
    """
    Simple bisection method. Finds zero of L between x_left and x_right.
    """
    sign_left = sign(L(x_left))
    sign_right = sign(L(x_right))

    assert sign_left != sign_right
    
    while x_right - x_left > 1e-10:
	x_middle = 0.5 * (x_left + x_right)
	sign_middle = sign(L(x_middle))
	if sign_middle == sign_left:
	    x_left = x_middle
	else:
	    x_right = x_middle
    
    return x_middle


def test_gausslaguerre(rule_order):
    """
    Testing the class Gauss_laguerre_quadrature_rule.
    

    Not enthusiastic about these results. The x**n is not good.
    Only good up to rule order 16.
    """
	
    my_rule = quad.Gauss_laguerre_quadrature_rule(rule_order)
    
#    X = my_rule.nodes
#    W = my_rule.weights
#    
#    L = genlaguerre(rule_order + 1,0)
#
#
#    W_2 = X/((rule_order + 1)*L(X))**2
#    #W_2 = X/(L(X))**2
#    E = 0
##    E = (factorial(rule_order))**2./factorial(2. * rule_order)
#
#    return  X, W, W_2, E
#


    print "Integrating test functions:"
    
    #integral from zero to infinity of 
    #	exp(-x) x**n = n!
    for i in range(100):
	f = lambda x: x**i
	quad_answer = my_rule.integrate(f)
	exact_answer = factorial(i)
	#print "%1.5f"%exact_answer, "\t", "%1.5f"%(quad_answer)
	print "x**%i\t:"%i,
	print "Error = ", quad_answer - exact_answer, "\t Relative Error = ",(
		quad_answer - exact_answer)/exact_answer


#    #integral from zero to infinity of 
#    #other functions...
#
#    values = linspace(0.01, 0.99, 10) 
#    for i in values:
#	f = lambda x: 1./(i*exp(-x) - 1.)
#	quad_answer = my_rule.integrate(f)
#	exact_answer = -1./i * log(-1./(i - 1))
#	print exact_answer
#	print "1/(%0.3f * exp(-x) - 1)"%i,
#	print "Error = ", quad_answer - exact_answer, "\t Relative Error = ",(
#		quad_answer - exact_answer)/exact_answer
#
#    f = lambda x:log(x)
#    quad_answer = my_rule.integrate(f)
#    exact_answer = -0.577215665
#    print exact_answer
#    print "ln(x)",
#    print "Error = ", quad_answer - exact_answer, "\t Relative Error = ",(
#	    quad_answer - exact_answer)/exact_answer
#
#
#    values = linspace(0.01, 10, 10) 
#    for i in values:
#	f = lambda x: cos(i*x)
#	quad_answer = my_rule.integrate(f)
#	exact_answer = 1/(1. + i**2)
#	print exact_answer
#	print "sin(%2.2f * x)"%i,
#	print "Error = ", quad_answer - exact_answer, "\t Relative Error = ",(
#		quad_answer - exact_answer)/exact_answer
#


def test_hidden_quantum_number(psi, basis):
    """
    Looking at each variable function of the wavefunction.
    """
    #1st suspect: eta part.
    eta_grid = linspace(-1,1,1000)
    eta_fun = zeros(shape(eta_grid))
    for i, [m, nu, mu] in enumerate(basis.index_iterator):
	eta_fun += basis.evaluate_V(eta_grid, m, mu) * psi[i]

    #figure()
    #plot(eta_grid, eta_fun)
    #show()

    return eta_grid, eta_fun

