"""
This script runs the H2+ software, and creates HDF5 files containing the 
dipole matrix, as well as the eigenstates (electronic and vibrational) for 
analysis purposes.

The problem is defined by the parameters in the first three methods, and is 
solved in parallel by writing this in the terminal:

$ mpirun -n X python main.py

(Exchange X with the number of processors to be used. Should be > 1.)
"""

from numpy import r_, linspace, array
#HACK, SIGURD
import sys
sys_path_temp = ["/work/shared/uib/sas044/python_modules/lib/python2.7/site-packages"]
for i in sys.path:
    sys_path_temp.append(i)

sys.path = sys_path_temp
#####


#Hack
import os
os.environ['HOME'] = "/work/sas044"

import matplotlib
matplotlib.use("agg")
#
import electronic_BO
import tdse_electron
import vibrational_BO
import config
import name_generator as name_gen

import pypar


def el_tise_problem_parameters():
    """
    m_max, nu_max, mu_max, R_grid, beta, theta = el_tise_problem_parameters()

    This is the place to choose the problem parameters of the electronic TISE
    problem.
    """
    #Max of the m quantum number.
    m_max = 0

    #Max of the 'xi' quantum number.
    nu_max = 50

    #Max of the 'eta' quantum number.
    mu_max = 20

    #The internuclear distances one wishes to evaluate TISE for.
    R_grid = linspace(0.4, 20, 99)

    #The modulus of the complex scaling.	
    beta = 1.0

    #The argument of the complex scaling.
    theta = 0.0
    
    return m_max, nu_max, mu_max, R_grid, beta, theta


def el_tdse_problem_parameters():
    """
    filename, m, q, E_lim = el_tdse_problem_parameters()

    This is the place to choose the problem parameters of the electronic TDSE
    problem.
    """
    #The m values to be included in the TDSE basis.
    m = [0]

    #The q values to be included in the TDSE basis.
    q = [0,1,2,3]

    #The highest energy states to include (for R = 2).
    E_lim = 4
   
    #Name of the file where the electronic states are stored.
    #----------------
    #Independent problem:
    #filename = "el_states_m_0_nu_25_mu_15_beta_1_00_theta_0_00.h5"
    #-------------
    #Generate name from info in el_tise_problem_parameters().
    m_max, nu_max, mu_max, R_grid, beta, theta = el_tise_problem_parameters()
    
    conf = config.Config(m = m_max, nu = nu_max, mu = mu_max, 
	R = R_grid[0], beta = beta, theta = theta)
    
    filename = name_gen.electronic_eigenstates_R(conf)
    #-------------

    return filename, m, q, E_lim 


def vib_problem_parameters():
    """
    filename_el, nr_kept, xmin, xmax, xsize, order = vib_problem_parameters()

    This is the place to choose the problem parameters of the vibrational
    problem.
    """
    
    #Number of vibrational states to be kept when diagonalizing.
    nr_kept = 200

    #Start of the R grid.
    xmin = 0.5
    
    #End of the R grid.
    xmax = 18.5

    #Approximat number of B-splines.
    xsize = 399

    #Order of the B-splines.
    order = 5
    
    #Name of the file where the electronic couplings are stored.
    #--------------------
    #Independent problem:
    #filename_el = ("el_couplings_lenght_z_m_0_nu_25_mu_15_beta_1_00_" + 
    #	"theta_0_00_m_0_q_3_E_5_00.h5")
    #-------------------- 
    #Generate name from info in el_tise_problem_parameters().
    m_max, nu_max, mu_max, R_grid, beta, theta = el_tise_problem_parameters()
    
    conf = config.Config(m = m_max, nu = nu_max, mu = mu_max, 
	R = R_grid[0], beta = beta, theta = theta)
    
    tdse_instance = tdse_electron.TDSE_length_z(conf = conf)

    filename, m_list, q_list, e_lim = el_tdse_problem_parameters()
    
    filename_el = name_gen.electronic_eig_couplings_R(tdse_instance, 
	m_list, q_list, e_lim)
    #--------------------

    return  filename_el, nr_kept, xmin, xmax, xsize, order



def main():
    
#    #=========================================================================
#    #==============    Electronic TISE   =====================================
#    #=========================================================================
#    #Get parameters.
#    m_max, nu_max, mu_max, R_grid, beta, theta = el_tise_problem_parameters()
#   
#    #Do calculations.
#    electronic_BO.save_electronic_eigenstates(m_max, nu_max, mu_max, R_grid, beta, theta)
#
#    #=========================================================================   
#    #==============    Electronic TDSE   =====================================
#    #=========================================================================
#    #Get parameters.
#    filename, m, q, E_lim = el_tdse_problem_parameters()
#    
#    #Do calculations.
#    tdse = tdse_electron.TDSE_length_z(filename = filename)
#    tdse.BO_dipole_couplings(m, q, E_lim)
#
#    #=========================================================================   
#    #==============    Vibrational TISE   ====================================
    #=========================================================================
    #Get problem parameters.
    filename_el, nr_kept, xmin, xmax, xsize, order = vib_problem_parameters()
    
    #Do calculations.
    vibrational_BO.save_all_eigenstates(filename_el, nr_kept, 
	xmin, xmax, xsize, order)
    
    #=========================================================================   
    #==============    Vibrational TDSE   ====================================
    #=========================================================================
    #Get problem parameters.
    filename_el, nr_kept, xmin, xmax, xsize, order = vib_problem_parameters()
    
    #Do calculations.
    vibrational_BO.save_all_couplings(filename_el, nr_kept, 
    	xmin, xmax, xsize, order)
    
    #Calculate couplings for the eigenfunction basis.
    #vibrational_BO.save_eigenfunction_couplings(filename_el, nr_kept, 
    #	xmin, xmax, xsize, order)
    
    #=========================================================================
   
    pypar.finalize()

#Run the main method.
main()
