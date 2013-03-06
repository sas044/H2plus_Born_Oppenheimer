"""
Creates the Franck-Condon wavepacket in a b-spline basis.
"""
import sys
sys.path.append("..")

import numpy
from numpy import *
import vibrational_methods
import tables


def make_fc_init(filename, filename_out, nr_el, nr_vib):
    """
    make_fc_init(filename, filename_out, nr_el, nr_vib)

    Creates the Franck-Condon wavepacket in a b-spline basis. 
    """
    r   = numpy.load("r_0.npy")
    p_r = numpy.load("p_0_r.npy")
    p_i = numpy.load("p_0_i.npy")

    f = tables.openFile(filename)
    
    xmin  = float(f.root.splineInfo[0])
    xmax  = float(f.root.splineInfo[1])
    xsize = int(f.root.splineInfo[2])
    order = int(f.root.splineInfo[3])

    f.close()

    spline = vibrational_methods.Bspline_basis(xmin,xmax,xsize,order)
    spline.setup_overlap_matrix()
	
    c_r = spline.bspline_expansion(r, p_r)
    c_i = spline.bspline_expansion(r, p_i)
    
    #Padding. Assumes the wavepackage should be in the electronic ground state.
    
    init = zeros([2 * nr_el * nr_vib])
    
    init[:len(c_r)] = c_r
    init[nr_el * nr_vib:nr_el * nr_vib + len(c_i)] = c_i
    
    f = tables.openFile(filename_out, "w")
    f.createArray("/", "initial_state", init)
    f.close()



