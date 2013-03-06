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
#=============================================================================
#Propagation parameters:    --------------------------------------------------
#=============================================================================

#Basis
#-----
#Time independent hamiltonian.
q_val = 1
name_ti = "../Data/vib_states_m_0_q_%i_xmin_0_50_xmax_18_50_size_4%i0_order_5.h5"%(q_val,q_val)
#Time dependent hamiltonian.
name_td =  "../Data/couplings_m_0_q_%i_xmin_0_50_xmax_18_50_size_4%i0_order_5.h5"%(q_val,q_val)


##Time independent hamiltonian.
#name_ti = "../Data/vib_states_m_0_q_3_xmin_0_50_xmax_18_50_size_401_order_5.h5"
##Time dependent hamiltonian.
#name_td =  "../Data/couplings_m_0_q_3_xmin_0_50_xmax_18_50_size_401_order_5.h5"

#Name of file containing the initial function.
name_init = "FC_initial_state_delay_0.h5"

#Laser
#-----
#amplitude =  0.25
#omega =  0.8471
#cycles = 6
#extra_time =  0.02

amplitude =  0.25
omega =  0.8471
cycles = 6
extra_time =  0.01

#Saving
#------
time_steps = 1000

#Initial state
#-------------
el_value = 1
vib_value = 1

#Restart
#-------
#1 : no restart
#>1 : restart job from the given index.
start_index = 1

#Job info
#--------
#Job duration.
hours = 1
minutes = 0
#Memory per process.
memory = 1000

#=============================================================================
#-----------------------------------------------------------------------------
#=============================================================================

import tables
import numpy
import os

#Create input file for the fortran routine. 
#----
#Output filename.
dir_name = name_td.split("/")[-1][10:-3]
if dir_name not in os.listdir("output"):
    os.system("mkdir output/%s"%dir_name)

filename_out = "output/%s/amplitude_%2.5f_omega_%2.3f_cycles_%i_extra_time_%2.2f.txt"%(
dir_name, amplitude, omega, cycles, extra_time)


f = open("input.txt","w")
#Line 1 - output filename.
f.write("\"%s\"\n"%filename_out)
#Line 2 - laser amplitude.
f.write("%f\n"%amplitude)
#Line 3 - laser frequency.
f.write("%f\n"%omega)
#Line 4 - laser cycles.
f.write("%i\n"%cycles)
#Line 5 - additional tima after laser.
f.write("%f\n"%extra_time)
#Line 6 - time steps.
f.write("%i\n"%time_steps)
#Line 7 - start index.
f.write("%i\n"%start_index)
#Line 8 - el value.
f.write("%i\n"%el_value)
#Line 9 - vib value.
f.write("%i\n"%vib_value)
#Line 10 - basis file, time independent.
f.write("\"%s\"\n"%name_ti)
#Line 11 - basis file, time dependent.
f.write("\"%s\"\n"%name_td)
#Line 12 - name of hdf5 with initial wavefunction.
f.write("\"%s\"\n"%name_init)
f.close()


#Make and execute the job file for the fortran routine.
#----
#Get number of electronic states.
f = tables.openFile(name_ti)
procs = f.root.hamiltonian.shape[2]
f.close()

procs_per_node = 32000/memory

f = open('job_script','w')
f.write('#!/bin/bash\n')
f.write('#PBS -N "H2plus"\n')
f.write('#PBS -A fysisk\n')
f.write('#PBS -l walltime=%02i:%02i:00\n'%(hours, minutes))
f.write('#PBS -l mppwidth=%i\n'%(procs))
f.write('#PBS -l mppmem=%imb\n'%memory)
f.write('#PBS -l mppnppn=%i\n'%procs_per_node)
f.write('cd /work/sas044/BO_H2plus/Fortran_propagation\n')
f.write('aprun -B ./main')
f.close()

#os.system("qsub job_script")


