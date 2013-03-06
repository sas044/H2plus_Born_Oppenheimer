"""
Script for submitting a job on the supercomputer Hexagon.
"""
import os


hours = 1
minutes = 0
procs = 99
memory = 1000

procs_per_node = 32000/memory

f = open('job_script','w')
f.write('#!/bin/bash\n')
f.write('#PBS -N "H2plus"\n')
f.write('#PBS -A fysisk\n')
f.write('#PBS -l walltime=%02i:%02i:00\n'%(hours, minutes))
f.write('#PBS -l mppwidth=%i\n'%(procs))
f.write('#PBS -l mppmem=%imb\n'%memory)
f.write('#PBS -l mppnppn=%i\n'%procs_per_node)
f.write('cd /work/sas044/new_BO/BO_H2plus\n')
f.write('aprun -B python main.py')
f.close()

os.system("qsub job_script")


