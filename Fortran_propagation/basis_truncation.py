from numpy import *
from pylab import *

import tables



class Truncation_config:
    """
    Class for uniquely defining a truncated basis.
    Contains storing and loading methods.
    """
    def __init__(self, old_vib_name = None, indices = None, tag = None, 
	config = None):
	"""
	config_object = Truncation_config( old_vib_name = None, 
	    indices = None, tag = None, config = None)

	Constructor. Supply naem of stored config object or the specs.

	Parameters
	----------
	old_vib_name : string, name of the HDF5 file containing the vibrational 
	    states of the basis we will truncate.
	indices : list, the indices of the electronic states in the original 
	    electronic basis file that will make the new basis.
	tag : string, identifier to avoid overwriting other bases.
	config : string, name of a config file, for recreation of the 
	    saved object.
	"""
	if config == None:
	    self.indices = indices
	    self.tag = tag
	    
	    #Assumes it is called from a subfolder of BO_H2plus.
	    f = tables.openFile("../Data/%s"%old_vib_name)
	    self.name_el_couplings_old = "../%s"%f.root.electronicFilename[0]
	    f.close()

	    self.name_vib_states_old = "../Data/%s"%old_vib_name
	    self.name_couplings_old = "../Data/couplings%s"%old_vib_name[10:]
	    
	    self.name_el_couplings = "%s_trunc_%s.h5"%(
		self.name_el_couplings_old[:-2], tag)
	    self.name_couplings = "%s_trunc_%s.h5"%(
		self.name_couplings_old[:-2], tag)
	    self.name_vib_states = "%s_trunc_%s.h5"%(
		self.name_vib_states_old[:-2], tag)
	else:
	    self.load_config(config)
    
    def save_config(self, storage_directory):
	"""
	save_config(storage_directory)

	Stores a config object.

	Parameter
	---------
	storage_directory : string, self explanatory.

	OBS:
	----
	All changes to this program must be mirrored in load_config().
	"""
	f = open("%s/info.txt"%storage_directory, "w")
	f.write("%s\n"%str(self.indices))
	f.write(self.tag + "\n")
	f.write(self.name_el_couplings_old + "\n")
	f.write(self.name_vib_states_old + "\n")
	f.write(self.name_couplings_old + "\n")
	f.write(self.name_el_couplings + "\n")
	f.write(self.name_vib_states + "\n")
	f.write(self.name_couplings)
	f.close()

    
    def load_config(self, config_path):
	"""
	load_config(config_path)

	Restores a saved config object. 

	Parameter
	---------
	config_path : string, the path of the config file.
	"""
	f = open(config_path)
	self.indices = eval(f.readline()[:-1])
	self.tag = f.readline()[:-1]
	self.name_el_couplings_old = f.readline()[:-1]
	self.name_vib_states_old = f.readline()[:-1]
	self.name_couplings_old = f.readline()[:-1]
	self.name_el_couplings = f.readline()[:-1]
	self.name_vib_states = f.readline()[:-1]
	self.name_couplings = f.readline()
	f.close()


    def equals(self, comp):
	"""
	is_equal = equals(comp)

	To compare if two truncated bases are the same.

	Parameter
	---------
	comp : truncation object.

	Returns
	-------
	is_equal : boolean, True if the objects are equal.
	"""

	if self.indices != comp.indices:
	    return False
	if self.tag != comp.tag:
	    return False
	if self.name_el_couplings_old != comp.name_el_couplings_old:
	    return False
	if self.name_vib_states_old != comp.name_vib_states_old:
	    return False
	if self.name_couplings_old != comp.name_couplings_old:
	    return False
	if self.name_el_couplings != comp.name_el_couplings:
	    return False
	if self.name_vib_states != comp.name_vib_states:
	    return False
	if self.name_couplings != comp.name_couplings:
	    return False
	return True	


	
def truncate_vib_states(old_filename, indices, tag):
    """
    new_filename = truncate_vib_states(old_filename, indices, tag)

    Creates a truncated version of the vibrational basis, using only the 
    electronic states given in <indices>.

    Parameters
    ----------
    old_filename : string, the name of the HDF5 file containing the 
	vibrational states and the time independent hamiltonian.
    indices : integer list, the electronic states selected from the states in 
	the HDF5 file of electronic couplings. 
    tag : string, identifier to avoid overwriting files.

    Returns
    -------
    new_filename : string, the name of the HDF5 file containing 
	the truncated basis.
    """
    
    new_filename = "%s_trunc_%s.h5"%(old_filename[:-3], tag)
    
    f = tables.openFile(old_filename)
    g = tables.openFile(new_filename, 'w')

    try:
	#Copying the small stuff.
	g.createArray("/", "electronicFilename", 
	    ["%s_trunc_%s.h5"%(f.root.electronicFilename[0][:-3], tag)])
	g.createArray("/", "overlap", f.root.overlap[:])
	g.createArray("/", "R_grid", f.root.R_grid[:])
	g.createArray("/", "splineInfo", f.root.splineInfo[:])
	g.createArray("/", "indices", indices)
	g.createArray("/", "full_name", [old_filename])
	

	#Initializing the arrays for the eigenvalues and states.
	#Number of el states included.
	nr_tasks = len(indices)
	
	g.createCArray('/','E', tables.atom.FloatAtom(), 
	    (f.root.E.shape[0], nr_tasks), chunkshape = f.root.E.chunkshape)
	
	g.createCArray('/','V', tables.atom.FloatAtom(), 
	    (f.root.V.shape[0], f.root.V.shape[1], nr_tasks),
	    chunkshape = f.root.V.chunkshape)
	
	g.createCArray('/','hamiltonian', tables.atom.FloatAtom(),    
	    (f.root.hamiltonian.shape[0], f.root.hamiltonian.shape[1], 
	    nr_tasks), chunkshape = f.root.hamiltonian.chunkshape)
	
	#Filling the large arrays.
	for i, ind in enumerate(indices):
	    g.root.E[:,i] = f.root.E[:,ind]
	    g.root.V[:,:,i] = f.root.V[:,:,ind]
	    g.root.hamiltonian[:,:,i] = f.root.hamiltonian[:,:,ind]

    finally:
	f.close()
	g.close()
    
    return new_filename

def truncate_couplings(old_filename, indices, tag):
    """
    new_filename = truncate_couplings(old_filename, indices, tag)

    Creates a truncated version of the time dependent Hamiltonian, 
    using only the electronic states given in <indices>.

    Parameters
    ----------
    old_filename : string, the name of the HDF5 file containing the 
	couplings of the time dependent Hamiltonian.
    indices : integer list, the electronic states selected from the states in 
	the HDF5 file of electronic couplings. 
    tag : string, identifier to avoid overwriting files.

    Returns
    -------
    new_filename : string, the name of the HDF5 file containing 
	the truncated Hamiltonian.
    """
    
    new_filename = "%s_trunc_%s.h5"%(old_filename[:-3], tag)
    
    f = tables.openFile(old_filename)
    g = tables.openFile(new_filename, 'w')

    try:
	#Copying the small stuff.
	g.createArray("/", "electronicFilename", 
	    ["%s_trunc_%s.h5"%(f.root.electronicFilename[0][:-3], tag)])
	g.createArray("/", "splineInfo", f.root.splineInfo[:])
	g.createArray("/", "indices", indices)
	g.createArray("/", "full_name", [old_filename])
	

	#Initializing the arrays for the eigenvalues and states.
	
	#Number of el states included.
	nr_el = len(indices)
	nr_vib = f.root.couplings.chunkshape[0]

	g.createCArray('/','couplings', tables.atom.FloatAtom(), 
	    (nr_el * nr_vib, nr_el * nr_vib), 
	    chunkshape = (nr_vib, nr_vib))
	
	#Filling the large arrays.
	for i_1, ind_1 in enumerate(indices):
	    for i_2, ind_2 in enumerate(indices):
		#Slices.
		g_1 = slice(i_1 * nr_vib, (i_1 + 1) * nr_vib) 
		g_2 = slice(i_2 * nr_vib, (i_2 + 1) * nr_vib) 
		f_1 = slice(ind_1 * nr_vib, (ind_1 + 1) * nr_vib) 
		f_2 = slice(ind_2 * nr_vib, (ind_2 + 1) * nr_vib) 

		#Copying cunks of the array.
		g.root.couplings[g_1, g_2] = f.root.couplings[f_1, f_2]

    finally:
	f.close()
	g.close()
    
    return new_filename

def truncate_el_couplings(old_filename, indices, tag):
    """
    new_filename = truncate_el_couplings(old_filename, indices, tag)

    Creates a truncated version of the time dependent electronic couplings, 
    using only the electronic states given in <indices>.

    Parameters
    ----------
    old_filename : string, the name of the HDF5 file containing the 
	electronic couplings of the time dependent Hamiltonian.
    indices : integer list, the electronic states selected from the states in 
	the HDF5 file of electronic couplings. 
    tag : string, identifier to avoid overwriting files.

    Returns
    -------
    new_filename : string, the name of the HDF5 file containing 
	the truncated electronic couplings.
    """
    
    new_filename = "%s_trunc_%s.h5"%(old_filename[:-3], tag)
    
    f = tables.openFile(old_filename)
    g = tables.openFile(new_filename, 'w')

    try:
	#Copying the small stuff.
	g.createArray("/", "index_array", f.root.index_array[indices,:])
	g.createArray("/", "config", f.root.config[:])
	g.createArray("/", "R_grid", f.root.R_grid[:])
	g.createArray("/", "indices", indices)
	g.createArray("/", "full_name", [old_filename])
	

	#Initializing the arrays for the eigenvalues and states.
	
	#Number of el states included.
	nr_el = len(indices)
	
	g.createCArray('/', 'E', tables.atom.FloatAtom(), 
	    (nr_el, f.root.E.shape[1]), chunkshape=(nr_el, 1))
	g.createCArray('/','couplings', tables.atom.FloatAtom(), 
	    (nr_el, nr_el, f.root.couplings.shape[2] ), 
	    chunkshape = (nr_el, nr_el, 1))
	
	#Filling the large arrays.
	for i_1, ind_1 in enumerate(indices):
	    for i_2, ind_2 in enumerate(indices):
		#Copying cunks of the array.
		g.root.couplings[i_1,i_2,:] = f.root.couplings[ind_1,ind_2,:]

    finally:
	f.close()
	g.close()
    
    return new_filename

