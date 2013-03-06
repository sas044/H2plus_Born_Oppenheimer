from numpy import *
from pylab import *

import tdse_electron
import psc_basis
import time
import tables


def test_dipole_matrix():
    """
    Test of the method dipole_psc().
    """

    filename = "el_states_m_0_nu_25_mu_15_beta_1_00_theta_0_00.h5"
    basis = psc_basis.PSC_basis(filename = filename)
    tdse = tdse_electron.TDSE_length_z(filename = filename)
    
    t_1 = time.time() 
    D = tdse.dipole_psc(basis)
    t_2 = time.time()
    print t_2 - t_1
    
    return D

def test_calculate_dipole_eig_R():
    R_index = 5

    filename = "el_states_m_0_nu_25_mu_15_beta_1_00_theta_0_00.h5"
    tdse = tdse_electron.TDSE_length_z(filename = filename)
    
    f = tables.openFile(tdse.coupling_file)
    try:
	index_array = f.root.index_array[:]
	R = f.root.R_grid[R_index]
    finally:
	f.close()
    t_1 = time.time() 
    D,E = tdse.calculate_dipole_eig_R(index_array, R)
    t_2 = time.time()
    print t_2 - t_1
    
    tdse.save_dipole_eig_R(D, E, R)

    return D, E

def test_coupling_interpolation(i = 200, j = 159):
    """
    Trying to automatically remove the points that are plainly wrong from the 
    electronic couplings.
    """
    def remove_point(c,i):
	if i == 0:
	    new_point = c[i+1]
	elif i == len(c) - 1:
	    new_point = c[i-1] 
	else:
	    new_point = (c[i-1] + c[i+1])/2.
	
	d = copy(c)
	d[i] = new_point
	return d

    f = tables.openFile(
	"el_couplings_lenght_z_m_0_nu_70_mu_25_beta_1_00_theta_0_00.h5")
    C = f.root.couplings[:]
    R = f.root.R_grid[:]
    f.close()
    c_0 = C[i,j,:]
    c_1 = C[i,j,:]
    dc = abs(diff(c_1))
    limit = 2 * average(dc) #3?
    indices = find(dc > limit)
    ii_2 = find(diff(indices) == 1)
    remove_these = indices[ii_2 + 1]
    
    for i in remove_these:
	c_1 = remove_point(c_1,i)
    
    plot(R,c_0,'--', R,c_1)
    title(len(remove_these))
    show()
    
    return R,c_0,c_1


    



def test_BO_dipole_couplings():
    """
    Test of the method BO_dipole_couplings().
    """

    filename = "el_states_m_0_nu_25_mu_15_beta_1_00_theta_0_00.h5"
    tdse = tdse_electron.TDSE_length_z(filename = filename)

    m = [0]
    q = [0,1,2,3]
    E_lim = 5.0

    tdse.BO_dipole_couplings(m, q, E_lim)
