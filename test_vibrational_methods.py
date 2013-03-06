import vibrational_methods as vib
import time

import tables
from numpy import *
from pylab import *

def test_H_kin_and_overlap():
    x_min = 0
    x_max = 30
    order = 7
    x_size = 350
    t_0 = time.time()
    basis = vib.Bspline_basis( x_min, x_max, x_size, order)
    t_1 = time.time()    
    basis.setup_kinetic_hamiltonian()
    t_2 = time.time()    
    basis.setup_overlap_matrix()
    t_3 = time.time()    
    
    print t_1 - t_0
    print t_2 - t_1
    print t_3 - t_2
    return basis


def map_to_index(task_nr, basis_size, version = "triangular"):
    """
    i, j = map_to_index(task_nr, basis_size, version = "triangular")

    Given a 1D index, the row and column index of a matrix element is found.

    Parameters
    ----------
    task_nr : integer, 1D index.
    basis_size : integer, length of the matrix.
    version : string, defaultly "triangular" (for hermitian matrices). 
	Otherwise, one assumes a full matrix.
    
    Returns
    -------
    i : integer, row index.
    j : iteger, column index.
    """
    if version == "triangular":
	i = 0
	while task_nr >= basis_size - i:	
	    task_nr -= basis_size - i
	    i += 1
	j = i + task_nr 
    else:
	#Row and column indices.
	i = task_nr / basis_size
	j = task_nr % basis_size

    return i,j

def test_couplings():
    """
    Checks some properties of the final H2+ hamiltonian.
    """
    f = tables.openFile("couplings_m_0_q_3_xmin_0_50_xmax_20_00_size_102_order_6.h5")
    C = f.root.couplings[:]
    f.close()

    symmetric = sum(C - transpose(C))
    print "Symmetric?", symmetric
    print "max:", max(C.ravel())
    print "min:", min(C.ravel())
    print "sum:", sum(C.ravel())

