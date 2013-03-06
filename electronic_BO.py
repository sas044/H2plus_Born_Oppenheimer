"""
Solving the electronic part of the H2+ problem in the Born Oppenheimer
approximation. This file contains the main method.
"""

from numpy import r_

import name_generator as name_gen
import config
import tise_electron
import nice_stuff
import tables
import pypar




def save_electronic_eigenstates(m_max, nu_max, mu_max, R_grid, beta, theta):
    """
    save_electronic_eigenstates(m_max, nu_max, mu_max, R_grid, beta, theta)

    This program solves the electronic TISE for a range of internuclear 
    distances, given in <R_grid>, and stores them in an HDF5 file. 
    This program must be run in parallel.
    
    Example
    -------
    To run this program on 5 processors:
    
	$  mpirun -n 5 python electronic_BO.py 
    """

    #Parallel stuff
    #--------------
    #Get processor 'name'.
    my_id = pypar.rank() 
    
    #Get total number of processors.
    nr_procs = pypar.size()
    
    #Get number of tasks.
    nr_tasks = len(R_grid)

    #Get a list of the indices of this processors share of R_grid. 
    my_tasks = nice_stuff.distribute_work(nr_procs, nr_tasks, my_id)

    #The processors will be writing to the same file.
    #In order to avoid problems, the procs will do a relay race of writing to
    #file. This is handeled by blocking send() and receive().
    #Hopefully there will not be to much waiting.

    #ID of the processor that will start writing.
    starter = 0

    #ID of the processor that will be the last to write.
    ender = (nr_tasks - 1) % nr_procs

    #Buffer for the baton, i.e. the permission slip for file writing.
    baton = r_[0]

    #The processor one is to receive the baton from.
    receive_from = (my_id - 1) % nr_procs 

    #The processor one is to send the baton to.
    send_to = (my_id + 1) % nr_procs 
    #-------------------------------

    
    #Initializing the HDF5 file
    #--------------------------
    if my_id == 0:
	#Creates a config instance.
	my_config = config.Config(m = m_max, nu = nu_max, mu = mu_max, 
	    R = R_grid[0], beta = beta, theta = theta)
	
	#Number of basis functions.
	basis_size = (2 * m_max + 1) * (nu_max + 1) * (mu_max + 1)

	#Generate a filename.
	filename = name_gen.electronic_eigenstates_R(my_config)

	f = tables.openFile(filename, 'w')
	try:
	    f.createArray("/", "R_grid", R_grid)	    
	    
	    #Looping over the m values.
	    for m in range(-1 * m_max, m_max + 1):
		#Creating an m group in the file.
		m_group = name_gen.m_name(m)
		f.createGroup("/", m_group)
		
		#Looping over th q values.
		for q in range(mu_max + 1):
		    #Creating a q group in the m group in the file.
		    q_group = name_gen.q_name(q)
		    f.createGroup("/%s/"%m_group, q_group)

		    #Initializing the arrays for the eigenvalues and states.
		    f.createCArray('/%s/%s/'%(m_group, q_group),'E', 
			tables.atom.FloatAtom(), 
			(basis_size/(mu_max + 1), nr_tasks),
			chunkshape=(basis_size/(mu_max + 1), 1))
		    
		    f.createCArray('/%s/%s/'%(m_group, q_group),'V', 
			tables.atom.ComplexAtom(16), 
			(basis_size, basis_size/(mu_max + 1), nr_tasks),
			chunkshape=(basis_size, basis_size/(mu_max + 1), 1))
	    
	finally:
	    f.close()
	
	#Save config instance.
	my_config.save_config(filename)
    #----------------------------------


    #Solving the TISE
    #----------------
    #Looping over the tasks of this processor.
    for i in my_tasks:
	#Creating TISE instance.
	tise = tise_electron.TISE_electron(m = m_max, nu = nu_max, 
	    mu = mu_max, R = R_grid[i], beta = beta, theta = theta)
	
	#Diagonalizing the hamiltonian.
	E,V = tise.solve()
	
	#First file write. (Send, but not receive baton.)
	if starter == my_id:
	    #Write to file.
	    tise.save_eigenfunctions_R(E, V, R_grid[i])

	    #Avoiding this statement 2nd time around.
	    starter = -1

	    #Sending the baton to the next writer.
	    pypar.send(baton, send_to, use_buffer = True)
	
	#Last file write. (Receive, but not send baton.)
	elif i == my_tasks[-1] and ender == my_id :
	    #Receiving the baton from the previous writer.
	    pypar.receive(receive_from, buffer = baton)

	    #Write to file.
	    tise.save_eigenfunctions_R(E, V, R_grid[i])
	
	#The rest of the file writes.
	else:
	    #Receiving the baton from the previous writer.
	    pypar.receive(receive_from, buffer = baton)

	    #Write to file.
	    tise.save_eigenfunctions_R(E, V, R_grid[i])

	    #Sending the baton to the next writer.
	    pypar.send(baton, send_to, use_buffer = True)
	
	
	#Showing the progress of the work.
	if my_id == 0:
	    nice_stuff.status_bar("Electronic BO calculations", 
		i, len(my_tasks))
    #----------------------------
    
    #Letting everyone catch up. 
    pypar.barrier()

    #Since the sign of the eigenfunctions are completely arbitrary, one must
    #make sure they do not change sign from one R to another.
    if my_id == 0:
	tise.align_all_phases()
    
    #Letting 0 catch up. 
    pypar.barrier()


