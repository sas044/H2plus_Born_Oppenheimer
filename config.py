from numpy import exp
import tables

class Config:
    """
    This class contains the relevant parameters of the PSC basis.
    The parameters are used as described in 
    Kamta and Bandrauk, Phys. Rev. A 71, 053407 (2005). 
    """

    def __init__(self, conf = None, filename = None, m = None, nu = None, mu = None, 
	    R = None, beta = None, theta = None):
	"""
	Constructor, initiating the config instance.

	Parameters
	----------
	    	    	    
	    Option I
	    --------
	    conf : Config instance, defining the problem.

	    Option II
	    --------
	    filename : string, name of HDF5 file, 
		in which a config instance is saved.
	    
	    Option III
	    ---------
	    m : integer, maximal value of the m quantum number.	
	    mu : integer, maximal value of the mu quantum number.	
	    nu : integer, maximal value of the nu quantum number.	
	    R : float, internuclear distance.	
	    beta : float, modulus of the complex scaling.	
	    theta : float, argument of the complex scaling.	

	"""
	#Option I
	#--------
	if conf != None:
	    #Copying the config object.
	    self.m_max  = conf.m_max
	    self.mu_max = conf.mu_max
	    self.nu_max = conf.nu_max
	    self.R      = conf.R
	    self.beta   = conf.beta
	    self.theta  = conf.theta
	    self.alpha  = conf.alpha

	#Option II
	#--------
	elif filename != None:
	    #Initializing variables.
	    self.load_config(filename)
	
	#Option III
	#---------
	else:
	    try:
		m*mu*nu*R*beta*theta
	    except (TypeError):
		print "All parameters have not been given a value."
		raise IOError
	
	    #Initializing variables.
	    self.m_max = m
	    self.mu_max = mu
	    self.nu_max = nu
	    self.R = R
	    self.beta = beta
	    self.theta = theta
	    self.alpha = beta * exp(1j * theta)

    def save_config(self, filename):
	"""
	save_config(filename)

	Saves the config parameters as a list in the HDF5 file <filename>.

	Parameters
	----------
	filename : string, name of HDF5 file.
	
	Notes
	-----
	Changes to this method must have a 
	corresponding change in load_config.
	"""
	#Collect parameters in a list (for storage).
	config_list = []
	config_list.append(self.m_max)
	config_list.append(self.mu_max)
	config_list.append(self.nu_max)
	config_list.append(self.R)
	config_list.append(self.beta)
	config_list.append(self.theta)

	f = tables.openFile(filename, 'a')
	try:
	    #Saving the list of parameters.
	    f.createArray("/", "config", config_list)
	finally:
	    f.close()

    def load_config(self,filename):
	"""
	load_config(filename)

	Given a HDF5 file, <filename>, this method extracts the config
	parameters from the file, and initializes the variables.
	
	Parameters
	----------
	filename : string, name of HDF5 file.
	"""
	
	f = tables.openFile(filename)
	#Assumes the config instance is saved as /root/config.
	try:
	    config_list = f.root.config[:]
	finally:
	    f.close()
	
	#Initializing variables.
	self.m_max   = int(config_list[0])
	self.mu_max  = int(config_list[1])
	self.nu_max  = int(config_list[2])
	self.R       = config_list[3]
	self.beta    = config_list[4]
	self.theta   = config_list[5]
	self.alpha = self.beta * exp(1j * self.theta)



