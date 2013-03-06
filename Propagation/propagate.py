"""
Script for propagating H2+, using the Python ode solver. 
Decent for small problems. Not reliable for big problems.
"""

from scipy.integrate import odeint
from numpy import zeros, pi, linspace, save, c_

import ode_support
import ode_support_parallel
import pypar
import tables
import os
import time
import numpy
import make_fc_init
#Time keeping.
t_0 = time.time()

#Laser parameters.
amplitude = 0.005338
omega = 0.317
cycles = 27

####OBS
size_r_grid = 3
######


####5 states
filename_vib =     "../Data/vib_states_m_0_q_2_xmin_0_50_xmax_30_00_size_501_order_5.h5"
filename_coupling = "../Data/couplings_m_0_q_2_xmin_0_50_xmax_30_00_size_501_order_5.h5"
#

##Franck-Condon state.
##--------------------
#f = tables.openFile("FC_initial_state_delay_0_q_2_big.h5")
#y_0 = f.root.initial_state[:]
#f.close()
#


#H2+ states.
#-----------
#Initial state, y_0. (Complex made 2xfloat.)
f = tables.openFile(filename_vib)
y_0 = zeros((2 * f.root.hamiltonian.shape[0] * f.root.hamiltonian.shape[2]))
el_index = 0
y_0[el_index * f.root.hamiltonian.shape[0] :
#1s sigma g (nu=3)
    (el_index + 1) * f.root.hamiltonian.shape[0]] = f.root.V[:,3,0]
#1s sigma g (nu=0)
#    (el_index + 1) * f.root.hamiltonian.shape[0]] = f.root.V[:,0,0]

f.close()


parallel = True
#parallel = False

if parallel: 
    #Time array.
    extra_time = 0.00
    times = linspace(0, 2. * pi * cycles/omega + extra_time, 300)

    #PARALLEL function input for odeint.
    input_function = ode_support_parallel.Ode_function_call(filename_coupling, filename_vib, 
	amplitude, omega, cycles, extra_time)
else:
    #Time array.
    extra_time = 0.06
    times = linspace(0, 2. * pi * cycles/omega + extra_time, 300)

    #Function input for odeint.
    input_function = ode_support.Ode_function_call(filename_coupling, filename_vib, 
	amplitude, omega, cycles)



########################################


#Solving the TDSE.
y, information = odeint(input_function.dpsi_dt, y_0, times, printmessg=1, full_output = 1)#, hmax = 0.03)


if pypar.rank() == 0:
    dir_name = filename_coupling.split("/")[-1][10:-3]
    if dir_name not in os.listdir("output"):
	os.system("mkdir output/%s"%dir_name)

    filename_out = "output/%s/R_%i_amplitude_%2.2f_omega_%2.2f_cycles_%i_extra_time_%2.2f"%(
    dir_name,size_r_grid, amplitude, omega,cycles, extra_time)

    save(filename_out, c_[times, input_function.time_function(times),y])

    t_1 = time.time()

    print "Runtime : ", (t_1 - t_0)/60., "min"
    print information


pypar.barrier()
pypar.finalize()
