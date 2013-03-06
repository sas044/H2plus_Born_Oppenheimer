"""
This file contains methods to generate names throughout the H2+ code.
"""
from numpy import sign
import tables

def m_name(m):
    """
    name = m_name(m)
    
    Name of a group in the HDF5 file where the eigenstates and 
    energies are stored.

    Parameters
    ----------
    m : integer, angular momentum projection quantum number.

    Returns
    -------
    name : string with the appropriate name.
    """
    #Since the hyphen/-/minus sign creates trouble as a group/node name in
    #the HDF5 file, a letter is used instead.
    prefixes = {-1:"n", 0:"", 1:"p"}

    #Constructing the name.
    name = "m_%s%i"%(prefixes[sign(m)], abs(m))

    return name

def q_name(q):
    """
    name = q_name(q)
    
    Name of a group in the HDF5 file where the eigenstates and 
    energies are stored.

    Parameters
    ----------
    q : integer, angular momentum projection quantum number.

    Returns
    -------
    name : string with the appropriate name.
    """    
    
    #Constructing the name.
    name = "q_%i"%(q)

    return name

def electronic_eigenstates(config):
    """    
    name = electronic_eigenstates(basis)
    
    Name of the HDF5 file where the eigenstates and 
    energies are stored.

    Parameters
    ----------
    config : Config instance, defining the problem.

    Returns
    -------
    name : string with the appropriate name.
    """
    #Getting the parameters on a convenient form.
    m = config.m_max
    nu = config.nu_max
    mu = config.mu_max
    
    #Left of the decimal point.
    R_int = int(config.R)
    beta_int = int(config.beta)
    theta_int = int(config.theta)
    
    #2 digits to the right of the decimal point.
    R_remainder = int(100 * (config.R - R_int))
    beta_remainder = int(100 * (config.beta - beta_int))
    theta_remainder = int(100 * (config.theta - theta_int))
    
    #Constructing the name.
    name = "Data/el_states_m_%i_nu_%i_mu_%i_R_%i_%02i_beta_%i_%02i_theta_%i_%02i.h5"%(
	m, nu, mu, R_int, R_remainder, 
	beta_int, beta_remainder, theta_int, theta_remainder)

    return name

def vibrational_eigenstates(filename_el, spline_basis):
    """    
    name = vibrational_eigenstates(filename_el, spline_basis)
    
    Name of the HDF5 file where the eigenstates and 
    energies are stored.

    Parameters
    ----------
    filename_el : string, name of file where the electronic couplings and 
	energies are stored.
    spline_basis : instance of the Bspline_basis class.
    
    Returns
    -------
    name : string with the appropriate name.
    """
    
    #Getting the parameters on a convenient form.

    f = tables.openFile(filename_el)
    try:
	index_array = f.root.index_array[:]
    finally:
	f.close()

    #Quantum numbert from the electronic part.
    m_max = max(index_array[:,0])
    q_max = max(index_array[:,1])
    
    #Spline info.
    x_min = spline_basis.bsplines.breakpoint_sequence[0]
    x_max = spline_basis.bsplines.breakpoint_sequence[-1]
    x_size = spline_basis.nr_splines
    order = spline_basis.spline_order

    #Left of the decimal point.
    x_min_int = int(x_min)
    x_max_int = int(x_max)
    
    #2 digits to the right of the decimal point.
    x_min_remainder = int(100 * (x_min - x_min_int))
    x_max_remainder = int(100 * (x_max - x_max_int))
    
    #Constructing the name.
    name = ("Data/vib_states_m_%i_q_%i_xmin_%i_%02i_"%(
	m_max, q_max, x_min_int, x_min_remainder)  +  
	"xmax_%i_%02i_size_%i_order_%i.h5"%(
	x_max_int, x_max_remainder, x_size, order))

    return name

def couplings(filename_el, spline_basis):
    """    
    name = couplings(filename_el, spline_basis)
    
    Name of the HDF5 file where the laser interaction hamiltonian is stored.

    Parameters
    ----------
    filename_el : string, name of file where the electronic couplings and 
	energies are stored.
    spline_basis : instance of the Bspline_basis class.
    
    Returns
    -------
    name : string with the appropriate name.
    """
    
    #Getting the parameters on a convenient form.

    f = tables.openFile(filename_el)
    try:
	index_array = f.root.index_array[:]
    finally:
	f.close()

    #Quantum numbert from the electronic part.
    m_max = max(index_array[:,0])
    q_max = max(index_array[:,1])
    
    #Spline info.
    x_min = spline_basis.bsplines.breakpoint_sequence[0]
    x_max = spline_basis.bsplines.breakpoint_sequence[-1]
    x_size = spline_basis.nr_splines
    order = spline_basis.spline_order

    #Left of the decimal point.
    x_min_int = int(x_min)
    x_max_int = int(x_max)
    
    #2 digits to the right of the decimal point.
    x_min_remainder = int(100 * (x_min - x_min_int))
    x_max_remainder = int(100 * (x_max - x_max_int))
    
    #Constructing the name.
    name = ("Data/couplings_m_%i_q_%i_xmin_%i_%02i_"%(
	m_max, q_max, x_min_int, x_min_remainder)  +  
	"xmax_%i_%02i_size_%i_order_%i.h5"%(
	x_max_int, x_max_remainder, x_size, order))

    return name
def eigenfunction_couplings(filename_el, spline_basis):
    """    
    name = couplings(filename_el, spline_basis)
    
    Name of the HDF5 file where the laser interaction hamiltonian is stored.

    Parameters
    ----------
    filename_el : string, name of file where the electronic couplings and 
	energies are stored.
    spline_basis : instance of the Bspline_basis class.
    
    Returns
    -------
    name : string with the appropriate name.
    """
    
    #Getting the parameters on a convenient form.

    f = tables.openFile(filename_el)
    try:
	index_array = f.root.index_array[:]
    finally:
	f.close()

    #Quantum numbert from the electronic part.
    m_max = max(index_array[:,0])
    q_max = max(index_array[:,1])
    
    #Spline info.
    x_min = spline_basis.bsplines.breakpoint_sequence[0]
    x_max = spline_basis.bsplines.breakpoint_sequence[-1]
    x_size = spline_basis.nr_splines
    order = spline_basis.spline_order

    #Left of the decimal point.
    x_min_int = int(x_min)
    x_max_int = int(x_max)
    
    #2 digits to the right of the decimal point.
    x_min_remainder = int(100 * (x_min - x_min_int))
    x_max_remainder = int(100 * (x_max - x_max_int))
    
    #Constructing the name.
    name = ("Data/eig_couplings_m_%i_q_%i_xmin_%i_%02i_"%(
	m_max, q_max, x_min_int, x_min_remainder)  +  
	"xmax_%i_%02i_size_%i_order_%i.h5"%(
	x_max_int, x_max_remainder, x_size, order))

    return name

def electronic_eigenstates_R(config):
    """    
    name = electronic_eigenstates_R(config)
    
    Name of the HDF5 file where the eigenstates and	    
    energies are stored.

    Parameters
    ----------
    config : Config instance, defining the problem.

    Returns
    -------
    name : string with the appropriate name.
    """
    
    #Getting the parameters on a convenient form.
    m = config.m_max
    nu = config.nu_max
    mu = config.mu_max
    
    #Left of the decimal point.
    beta_int = int(config.beta)
    theta_int = int(config.theta)
    
    #2 digits to the right of the decimal point.
    beta_remainder = int(100 * (config.beta - beta_int))
    theta_remainder = int(100 * (config.theta - theta_int))

    #Constructing the name.
    name = "Data/el_states_m_%i_nu_%i_mu_%i_beta_%i_%02i_theta_%i_%02i.h5"%(
	m, nu, mu, beta_int, beta_remainder, theta_int, theta_remainder)

    return name


def electronic_eig_couplings_R(tdse_instance, m_list, q_list, e_lim):
    """    
    name = electronic_eig_couplings_R(tdse_instance, m_list, q_list, e_lim)
    
    Name of the HDF5 file where the eigenstate dipole couplings are stored.

    Parameters
    ----------
    tdse_instance : TDSE_electron instance, defining the laser orientation and gauge.
    m_list : integer list, the m values included.
    q_list : integer list, the q values included.
    e_lim : float, the upper limit of the electronic energy for R = 2.

    Returns
    -------
    name : string with the appropriate name.
    """
    
    #Getting the parameters on a convenient form.
    tag = tdse_instance.laser_info
    m =  tdse_instance.config.m_max
    nu = tdse_instance.config.nu_max
    mu = tdse_instance.config.mu_max
    
    #Left of the decimal point.
    beta_int =  int(tdse_instance.config.beta)
    theta_int = int(tdse_instance.config.theta)
    e_lim_int = int(e_lim)
    
    #2 digits to the right of the decimal point.
    beta_remainder = int(100 * (tdse_instance.config.beta - beta_int))
    theta_remainder = int(100 * (tdse_instance.config.theta - theta_int))
    e_lim_remainder = int(100 * (e_lim - e_lim_int))

    #Truncations in m and q. Assumes that the list is uniquely given 
    #by its largest entry.
    m_max = max(m_list)
    q_max = max(q_list)



    #Constructing the name.
    name = ("Data/el_couplings_%s_m_%i_nu_%i_mu_%i_beta_%i_%02i_"%(  
	tag, m, nu, mu, beta_int, beta_remainder) +  
	"theta_%i_%02i_m_%i_q_%i_E_%i_%02i.h5"%(
	theta_int, theta_remainder, m_max, q_max, e_lim_int, e_lim_remainder))

    return name

