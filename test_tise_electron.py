"""
Testing battery to assure that the program 'tise_electron.py' is working properly.
"""
import tise_electron
import psc_basis
from numpy import *
from scipy.special import genlaguerre
from pylab import *
import time
import name_generator as name_gen
import tables

def test_hamiltonian():
    """
    Test of setup_hamiltonian().
    """
    t_1 = time.time()
    
    tise = tise_electron.TISE_electron(m = 0, nu = 25, mu = 20, 
	    R =2.0, beta = 1.0, theta= 0.0)
    E,V = tise.solve() 
    t_2 = time.time()
    tise.save_eigenfunctions(E,V)
    t_3 = time.time()
#    H = tise.hamiltonian
#    S = tise.basis.overlap_matrix

#    print "Symmetric? (0?)", sum(abs(ravel(H - transpose(H))))
#    print "Greatest error?", max(ravel(H - transpose(H)))
    print "Time (min):", (t_2 - t_1)/60 
    print "Time (min):", (t_3 - t_2)/60 
#    
#    print H[100,94], H[94,100]
#
#    h_prime = tise.find_h(-1, 3, 2)
#    h = tise.find_h(-1, 2, 3)
#    print "h", h_prime, h
#    
#    
#    d_prime = tise.basis.find_d(0,-1, 3, 2)
#    d = tise.basis.find_d(0,-1, 2, 3)
#    print "d", d_prime, d
    

#
#    figure()
#    pcolormesh(real(H))
#    colorbar()
#    show()
#
#    figure()
#    pcolormesh(real(H - transpose(H)))
#    colorbar()
#    show()
#    
    return tise, E, V

def test_align_all_phases():
    t_1 = time.time()
    
    tise = tise_electron.TISE_electron(m = 0, nu = 70, mu = 25, 
	    R =2.0, beta = 1.0, theta= 0.0)
    tise.align_all_phases() 
    t_2 = time.time()
    
    return 


def test_d_U():
    """
    When caluculating the matrix elements for the electronic hamiltonian, 
    or more specifically in eq. (A6) in Kamta2005, the differentiation of U 
    makes an appearance. Analytical formulas for dU and d2U are shown in the
    note A_6.pdf in the Documentation folder. This method checks the validity 
    of these formulas, using numerical differentiation for comparison.
    """
    basis = psc_basis.PSC_basis(m = 6, nu = 6, mu = 5, R = 2.0, beta = 0.8, theta= pi/6.)
    
    dxi = .05
    #An extra point, since one will be lost.
    xi = r_[1:10 + 2 * dxi:dxi]
    m = abs(4)
    nu = 5
    alpha = basis.config.alpha
    
    #Alternative variable.
    X = 2 * alpha * (xi - 1)    
    
    #Evaluation of U.
    U_num = basis.evaluate_U(xi, m, nu)

    #Analytical U
    U = basis.find_N(m,nu) * exp(-alpha * (xi - 1)) * (xi**2 - 1)**(abs(m)/2.) * genlaguerre(nu - abs(m), 2 * abs(m))(2 * alpha * (xi - 1))
    
    #single diff.
    #-------------------------------------------------------
    #Numerical differentiation.
    dU_num = (U_num[2:] - U_num[:-2])/(2. * dxi)

    L_0 = genlaguerre(nu - abs(m) + 0, 2 * abs(m))(2. * alpha * (xi-1))
    L_1 = genlaguerre(nu - abs(m) + 1, 2 * abs(m))(2. * alpha * (xi-1))
    L_2 = genlaguerre(nu - abs(m) + 2, 2 * abs(m))(2. * alpha * (xi-1))
    N = basis.find_N(m,nu)



    #Analytical differentiation, 2nd formula.
    dU = N * exp(- alpha * (xi - 1.)) * (xi**2. - 1.)**(abs(m)/2.) * (
	(alpha - abs(m)/(xi**2. - 1.) - (nu + 1.)/(xi-1)) * L_0 
	+ (nu - abs(m) + 1)/(xi - 1.) * L_1)


    #Analytical differentiation, X formula.
    dU_X = N * exp(-0.5 * X) * (X/(2* alpha))**(abs(m)/2.) * (X/(2* alpha) + 2)**(abs(m)/2.) * (
	(alpha - 4* alpha**2 * abs(m)/(X*(X + 4*alpha)) - 2* alpha*(nu + 1.)/X) * L_0 
	+ 2 * alpha * (nu - abs(m) + 1)/X * L_1)

    
    #double diff.
    #------------------------------------------------------
    #Numerical differentiation.
    d2U_num = (U_num[2:] - 2*U_num[1:-1] + U_num[:-2])/(dxi**2)
    
    #Diff of this : N * exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2) * [
    #	(alpha - m/(xi**2 - 1) - (nu + 1)/(xi - 1)) * L_0	
    #	+ (nu - m + 1)/(xi-1) * L_1
    #	]

#    #Analytical differentiation.
#    d2U = N * exp(-alpha * (xi - 1)) * (xi**2 - 1)**(m/2.) * (
#	( 
#	    alpha * (  
#		alpha
#		+ m * xi / (xi**2 - 1)
#		- (nu + m + 1) / (xi - 1))
#	    - m * ( 
#		alpha / (xi**2 - 1)
#		- ((nu + 3) * (xi + 1) - 2 + m)/(xi**2 - 1)**2)
#	    - (nu + 1) * ( 
#		alpha  / (xi - 1)
#		+ (m - 1)/(xi - 1)**2
#		- m / ((xi**2 - 1)*(xi - 1))
#		- (nu + m + 1) / (xi - 1)**2)
#	) * L_0 
#	+
#	(
#	    (nu - m + 1) * ( 
#	    alpha / (xi - 1)
#	    - m/((xi**2 - 1) * (xi - 1))
#	    - (nu + 1) / (xi - 1)**2
#	    + alpha  / (xi - 1)
#	    + (m - 1)/(xi - 1)**2
#	    - m / ((xi**2 - 1)*(xi - 1))
#	    - (nu + m + 2) / (xi - 1)**2 )
#	) * L_1
#	+ 
#	(
#	(nu - m + 1) * (nu - m + 2) / (xi - 1)**2	
#	) * L_2
#	)

    #Analytical differentiation.
    d2U_simplified = N * exp(-alpha * (xi - 1)) * (xi**2 - 1)**(m/2.) * (
	( 
	    (  
		alpha**2
		+ alpha * m / (xi + 1)
		- alpha * (2 * nu + m + 2) / (xi - 1))
	    + m * (m - 2)/ (xi**2 - 1)**2 
	    + (nu + 1) * (nu + 2) / (xi - 1)**2
	    + 2 * m * (nu + 2) / ((xi**2 - 1)*(xi - 1))
	) * L_0 
	+
	(
	    2 * (nu - m + 1) * ( 
	      alpha / (xi - 1)
	    - (nu + 2) / (xi - 1)**2
	    - m / ((xi**2 - 1)*(xi - 1)))
	) * L_1
	+ 
	(
	(nu - m + 1) * (nu - m + 2) / (xi - 1)**2	
	) * L_2
	)

    d2U_X = N * exp(-X/2) * (X/(2*alpha))**(m/2.) *(X/(2*alpha) + 2)**(m/2.) * (
	( 
	      
		alpha**2
		+ 2* alpha**2 * m / (X + 4*alpha)
		- 2 * alpha**2 * (2 * nu + m + 2) / X
	    + 16 * alpha**4 * m * (m - 2)/ (X**2* (X + 4*alpha)**2) 
	    + 4 * alpha**2 * (nu + 1) * (nu + 2) / X**2
	    + 16 * alpha**3 * m * (nu + 2) / (X**2*(X + 4*alpha))
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
	4 * alpha **2 * (nu - m + 1) * (nu - m + 2) / X**2	
	) * L_2
	)



    #Plotting result.
    figure()
    plot(X[1:-1], dU_num, X[1:-1], dU[1:-1], X[1:-1], dU_X[1:-1])
    show()

    figure()
    plot(X[1:-1], d2U_num, X[1:-1], d2U_X[1:-1], X[1:-1], d2U_simplified[1:-1] )
    show()
#
#    figure()
#    plot(xi, U_num, xi, U)
#    show()
#






def test_L0():
    basis = psc_basis.PSC_basis(m = 6, nu = 6, mu = 5, R = 2.0, beta = 0.8, theta= pi/6.)
    dxi = .05
    #An extra point, since one will be lost.
    xi = r_[1:10 + 2 * dxi:dxi]
    m = abs(4)
    nu = 5
    alpha = basis.config.alpha
    N = basis.find_N(m,nu)


    L_0 = genlaguerre(nu - abs(m) + 0, 2 * abs(m))(2. * alpha * (xi-1))
    L_1 = genlaguerre(nu - abs(m) + 1, 2 * abs(m))(2. * alpha * (xi-1))
    L_2 = genlaguerre(nu - abs(m) + 2, 2 * abs(m))(2. * alpha * (xi-1))

    
    
    #Part 1 : exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2) * alpha * L_0
    #----------------------------------------------------------
    #Numerical expression.
    U_num = exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2) * alpha * L_0

    #Numerical differentiation.
    dU_num = (U_num[2:] - U_num[:-2])/(2. * dxi)

    #Analytical differentiation.
    dU = exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2) * (
	(  alpha**2
	+ alpha * m * xi / (xi**2 - 1)
	- alpha * (nu + m + 1) / (xi - 1)
	) * L_0 
	+
	( alpha * (nu - m + 1) / (xi - 1)
	) * L_1
	)
    #OK!
    #----------------------------------------------------------	

    #Part 2 :  - exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2) * m/(xi**2 - 1) * L_0
    #----------------------------------------------------------
    #Numerical expression.
    U_num = - exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2) * m/(xi**2 - 1) * L_0

    #Numerical differentiation.
    dU_num += (U_num[2:] - U_num[:-2])/(2. * dxi)

    #Analytical differentiation.
#    dU = - m * exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2) * (
#	(
#	- alpha / (xi**2 - 1)
#	+ 2 * (m/2. - 1) * xi / (xi**2 - 1)**2
#	- (nu + m + 1 - 2* alpha * (xi - 1))/((xi**2 - 1) * (xi - 1))
#	) * L_0
#	+
#	(nu - m + 1)/((xi**2 - 1) * (xi - 1)) * L_1
#	)
   #Analytical differentiation, simplified.
    dU += - m * exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2) * (
	(
	+ alpha / (xi**2 - 1)
	- ((nu + 3) * (xi + 1) - 2 + m)/(xi**2 - 1)**2
	) * L_0
	+
	(nu - m + 1)/((xi**2 - 1) * (xi - 1)) * L_1
	)
    #OK!
    #----------------------------------------------------------

    #Part 3 :  - exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2) * (nu + 1)/(xi-1) * L_0
    #----------------------------------------------------------
    #Numerical expression.
    U_num = - exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2) * (nu + 1)/(xi-1) * L_0

    #Numerical differentiation.
    dU_num += (U_num[2:] - U_num[:-2])/(2. * dxi)

   #Analytical differentiation.
    dU += - (nu + 1) * exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2.) * (
	(
	+ alpha  / (xi - 1)
	+ (m - 1)/(xi - 1)**2
	- m / ((xi**2 - 1)*(xi - 1))
	- (nu + m + 1) / (xi - 1)**2 
	) * L_0
	+ (nu - m + 1) / (xi - 1)**2 * L_1
	)
    #OK!
    #----------------------------------------------------------
    
    #Part 4 :  exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2) * (nu - m + 1)/(xi-1) * L_1
    #----------------------------------------------------------
    #Numerical expression.
    U_num = exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2) * (nu -m + 1)/(xi-1) * L_1

    #Numerical differentiation.
    dU_num += (U_num[2:] - U_num[:-2])/(2. * dxi)

   #Analytical differentiation.
    dU += (nu - m + 1) * exp(-alpha*(xi - 1)) * (xi**2 - 1)**(m/2.) * (
	(
	+ alpha  / (xi - 1)
	+ (m - 1)/(xi - 1)**2
	- m / ((xi**2 - 1)*(xi - 1))
	- (nu + m + 2) / (xi - 1)**2 
	) * L_1
	+ (nu - m + 2) / (xi - 1)**2 * L_2
	)
    #OK!
    #----------------------------------------------------------

    #Analytical differentiation.
    d2U = N * exp(-alpha * (xi - 1)) * (xi**2 - 1)**(m/2.) * (
	( 
	    alpha * (  
		alpha
		+ m * xi / (xi**2 - 1)
		- (nu + m + 1) / (xi - 1))
	    - m * ( 
		alpha / (xi**2 - 1)
		- ((nu + 3) * (xi + 1) - 2 + m)/(xi**2 - 1)**2)
	    - (nu + 1) * ( 
		alpha  / (xi - 1)
		+ (m - 1)/(xi - 1)**2
		- m / ((xi**2 - 1)*(xi - 1))
		- (nu + m + 1) / (xi - 1)**2)
	) * L_0 
	+
	(
	    (nu - m + 1) * ( 
	    alpha / (xi - 1)
	    - m/((xi**2 - 1) * (xi - 1))
	    - (nu + 1) / (xi - 1)**2
	    + alpha  / (xi - 1)
	    + (m - 1)/(xi - 1)**2
	    - m / ((xi**2 - 1)*(xi - 1))
	    - (nu + m + 2) / (xi - 1)**2 )
	) * L_1
	+ 
	(
	(nu - m + 1) *(nu - m + 2) / (xi - 1)**2	
	) * L_2
	)




    #Plotting result.
    figure()
    plot(xi[1:-1], N * dU_num, xi[1:-1], N * dU[1:-1],xi[1:-1], d2U[1:-1] )
    show()

def test_ymse():
    x = linspace(0,10,1000)
#    u = 2*(x-1)
#    f = x * sin(x)
#    g = (0.5 * u + 1) * sin(0.5*u+1)
#    
#    figure()
#    plot(u,f,u,g)
#    show()
    
#    result = r_[
#    25.487925546,1  x 11
#    25.5036724957,15
#    25.503819671,25
#    25.5038703065,35
#    25.5038945277,45
#    25.5039082785,55
#    25.5039329611,105
#    25.5039452452,155
#    25.5039954889,205
#
#    25.7467649805,1
#    25.7544487215,15
#    25.7545221491,25
#    25.7545474316,35
#    25.7545595299,45
#    25.7545664005,55
#    25.7545789236,105
#    25.7545879198,155
#    25.7546367208,205

def test_extract_q():
    
    mu_max = 8

    
    tise = tise_electron.TISE_electron(m = 0, nu = 25, mu = mu_max, 
	    R = 14.170707070707, beta = 1.0, theta= 0.0)
    
    E,V = tise.solve() 
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

    
    figure()
    for j in range(15):
	title(j)
	plot(eta_grid, eta_fun[:,j])
    
    show()

    return tise, eta_grid, eta_funs, eta_fun
