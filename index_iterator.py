class Index_iterator():
    """
    This iterator class iterates over the 
    parameter permutations in the PSC basis.
    """
    def __init__(self, config):
	"""
	Index_iterator(config)

	Constructor. Makes <config> a object variable.
	
	Parameters
	----------
	config : a Config object, which contains the parameters needed 
	    to make the iterator.
	"""
	self.config = config

    def __getitem__(self, index):
	"""
	m, nu, mu = __getitem__(index)

	Retrieves the quantum numbers of function <index> in the basis.
	
	Parameters
	----------
	index : integer, the basis function whose quantum numbers you want.
	
	Returns
	----------
	m : integer, the angular momentum projection quantum number.
	nu : integer, some sort of quantum number associated with U.
	mu : integer, some sort of quantum number associated with V.
	"""
	
	counter = 0	
	
	#m : [-m_max, -m_max + 1,  ......, m_max - 1, m_max]
	for m in range(-1 * self.config.m_max, self.config.m_max + 1):
	    
	    #nu : [|m|, |m| + 1, ....., |m| + nu_max]
	    for nu in range(abs(m), abs(m) + self.config.nu_max + 1):
		
		#mu : [|m|, |m| + 1, ....., |m| + mu_max]
		for mu in range(abs(m), abs(m) + self.config.mu_max + 1):
		    
		    if counter == index:
			return m, nu, mu
		    else:
			counter += 1
	
	error_message = "Index out of bounds. "
	error_message += "Should be an integer in [0, basis_size - 1]."
	raise IndexError(error_message)

    def __iter__(self):
	"""
	Yields m, nu and mu in the correct order, according to how the basis 
	functions are organized. See Kamta2005, between eq. (8) and (9).
	"""

	#m : [-m_max, -m_max + 1,  ......, m_max - 1, m_max]
	for m in range(-1 * self.config.m_max, self.config.m_max + 1):
	    
	    #nu : [|m|, |m| + 1, ....., |m| + nu_max]
	    for nu in range(abs(m), abs(m) + self.config.nu_max + 1):
		
		#mu : [|m|, |m| + 1, ....., |m| + mu_max]
		for mu in range(abs(m), abs(m) + self.config.mu_max + 1):
		    
		    yield m, nu, mu 


