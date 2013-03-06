from pylab import reshape, find, shape
from numpy import vdot, dot, array, zeros, linspace, argmin
from numpy import real, imag, diff, intersect1d, ravel
from scipy.interpolate import UnivariateSpline

import os
import sys
sys.path.append("..")
import bsplines
import vibrational_methods
import tables
import config
import name_generator as name_gen
import numpy
import make_animation

class Analysis():
    """
    Class for analysing the dynamics of the BO-model of H2+.
    Provides methods for finding the energy spectrum and the
    probability density. Eventually it will make a movie of the dynamics.

    Note
    ----
    Assumes the wavefunction is stored in a particular fashion. 
    See retrieve_result(), or the propagation programs in ../Propagation/ 
    or ../Fortran_propagation.
    """

    def __init__(self, path_to_output_dir):
	"""
	Analysis(path_to_output_dir)

	Constructor. Makes an instance of the Analysis class.

	Parameters
	----------
	path_to_output_dir : string, the path to the relevant 
	    output directory.
	"""
	#Remove trailing slash.
	if len(path_to_output_dir.split("/")[-1]) == 0:
	    path_to_output_dir = path_to_output_dir[:-1]
	
	self.path_to_output_dir = path_to_output_dir
	#Name of the file where the vibrational things are stored.
	self.name_vib_states = "../Data/vib_states_%s.h5"%(
	    path_to_output_dir.split("/")[-1]) 
	
	#Name of file where the electronic couplings are.
	f = tables.openFile(self.name_vib_states)
	self.name_el_couplings = "../" + f.root.electronicFilename[0]
	self.overlap_matrix = f.root.overlap[:]
	self.spline_info = f.root.splineInfo[:]
	f.close()
	
	#Name of the file where the electronic states are.
	el_config = config.Config(filename = self.name_el_couplings)
	self.name_el_states = "../" + name_gen.electronic_eigenstates_R(
	    el_config)
	
	#Loading the vibrational B-spline basis.	
	self.vib_methods = vibrational_methods.Bspline_basis(
	    float(self.spline_info[0]), float(self.spline_info[1]), 
	    int(self.spline_info[2]), int(self.spline_info[3]))
	self.vib_basis = bsplines.load_spline_object(self.name_vib_states)
	
	#Basis sizes.
	f = tables.openFile(self.name_el_couplings)	    
	self.el_basis_size = f.root.index_array.shape[0]
	self.index_array = f.root.index_array[:]
	f.close()

	self.vib_basis_size = self.vib_basis.number_of_bsplines
	self.basis_size = self.vib_basis_size * self.el_basis_size
	self.psi_final = None
	self.splines = None
	self.times = None
	self.field = None
	
	#Continuum.
	self.find_continuum()
    	
    def retrieve_result(self, filename):
	"""
	retrieve_many_result(filename)

	Import the result of the dynamics as a class variable.

	Parameters
	----------
	filename : name of the result file.
	"""
	
	if filename[-3:] == "txt":
	    raw_result = numpy.loadtxt("%s/%s"%(self.path_to_output_dir, filename))
	    numpy.save("%s/%s"%(self.path_to_output_dir, filename[:-4]), raw_result)
	else:
	    raw_result = numpy.load("%s/%s"%(self.path_to_output_dir, filename))
	
	self.times = raw_result[:,0]
	self.field = raw_result[:,1]
	
	raw_result = raw_result[:,2:]


	self.psi = zeros([self.vib_basis_size, self.el_basis_size, 
	    raw_result.shape[0]], dtype = complex)
	
	for index in range(raw_result.shape[0]):
	    self.psi[:,:,index] = reshape(raw_result[index, :self.basis_size] + 
		1j * raw_result[index, self.basis_size:], 
		[self.vib_basis_size, self.el_basis_size], order = "F")
	self.psi_final = self.psi[:,:,-1]
    
    def ground_state_population(self, psi):
	"""
	population = ground_state_population(psi)

	Finds the population in the ground state.

	Parameters
	----------
	psi : 2D complex array, the wavefunction.

	Returns
	-------
	population : float, the population in the ground state.
	"""
	#File with ground state.
	f = tables.openFile(self.name_vib_states)

	#Doing the projection.
	try:
	    population = abs(self.inner_product(f.root.V[:,0,0], psi[:,0]))**2
	finally:
	    f.close()
	
	return population
    
    def named_state_population(self, psi, name_list):
	"""
	populations = named_state_population(psi, name_list)

	Returns the population in a set of named states.

	Parameters
	----------
	psi : 2D complex array. The wavefunction to be analyzed.
	name_list : string list, e.g. ["2p", "3d"].

	Returns
	-------
	populations : 1D float array, the population of the basis states. 
	"""

	angular_names = {"s":0, "p":1, "d":2, "f":3}

	el_indices = []

	for i in name_list:
	    q_matches = find(self.index_array[:,1] == (angular_names[i[1]]))
	    n_matches = find(self.index_array[:,2] == int(i[0]) - angular_names[i[1]] -1)
	    el_indices.append(intersect1d(q_matches, n_matches))
	
	el_indices = list(ravel(array(el_indices)))
	populations = self.state_population(psi, el_indices)

	return populations

    def state_population(self, psi, el_indices):
	"""
	populations = state_population(psi, el_indices)
	
	Returns the population in the wanted basis functions.
	
	Parameters
	----------
	psi : 2D complex array. The wavefunction to be analyzed.
	el_indices : integer list. The indices of the electronic functions 
	    for which to look at the population.  
	
	Returns
	-------
	populations : 1D float array, the population of the basis states.
	"""
	f = tables.openFile(self.name_vib_states)

	population = []
	try:
	    for index in el_indices:
		for j in range(f.root.V.shape[1]):
		    temp_pop = abs(self.inner_product(
			f.root.V[:,j,index], psi[:,index]))**2
		    population.append(temp_pop)    	
	finally:
	    f.close()
	
	population = array(population)
	
	return population

    def find_continuum(self):
	"""
	Creates a list of el states that are in the continuum.
	Stores the list as an instance variable.
	"""
	f = tables.openFile(self.name_el_couplings)
	
	#Electronic basis state energies.
	r_grid = f.root.R_grid[:]
	index_middle = argmin(abs(r_grid - 4))
	energies = f.root.E[:,index_middle]
	f.close()
	
	continuum = []
	
	#Looping the q's.
	for i in range(max(self.index_array[:,1])+1):

	    #Get the indices corresponding to  given q.
	    temp_ind = find(self.index_array[:,1] == i)
	    
	    #Corresponding energies.
	    temp_E = energies[temp_ind]
	    
	    #Checks if there are positive energies.
	    if temp_E[-1] > 0:
		#The continuum starts when gap between energies increases.
		#OBS: TODO DEBUG
		border_value = temp_ind[argmin(diff(temp_E))] + 0
		continuum.append(temp_ind[find((temp_ind > border_value))])
	
	self.continuum = [item for sublist in continuum for item in sublist]
	


    def ionization_probability(self, psi):
	"""
	prob = ionization_probability()

	Returns the probability of being in the continuum.

	Parameters
	----------
	psi : 2D complex array. The wavefunction to be analyzed.
	
	Returns
	-------
	prob : float, ionization probability.
	"""

	prob = 0
	
	#Find population in the states with an energy in R_max 
	#exceeding <limit>,
	for i in self.continuum:
	    prob += abs(real(self.inner_product(psi[:,i], psi[:,i])))
	
	return prob
    
    def ionized_energy_spectrum(self, psi, energy_grid):
	"""
	energy_grid, spectrum = ionized_energy_spectrum(psi, energy_grid, limit = 0)
	
	Returns the energy spectrum of the ionized wavefunction.
	
	Parameters
	----------
	psi : 2D complex array. The wavefunction to be analyzed.
	energy_grid : 1D float array.

	Returns
	-------
	energy_grid : 1D float array.
	spectrum : 1D float array, the energy spectrum of the wavefunction.
	"""

	energy_grid, spectrum = self.energy_spectrum(psi, self.continuum, energy_grid)

	return energy_grid, spectrum
    
    def projection_wavefunctions(self):
	"""
	E, V = projection_wavefunctions()

	Returns the vibrational wavefunctions corresponding to the 1/R 
	energy curve.

	Returns
	-------
	E : Vector containing the sorted eigenvalues.
	V : Array containing the normalized eigenvectors.
	"""
	r = linspace(0.1,100,1000)
	self.vib_methods.setup_overlap_matrix()
	self.vib_methods.setup_kinetic_hamiltonian()
	v = self.vib_methods.setup_potential_matrix(r, 1./r)
	E, V = self.vib_methods.solve(self.vib_methods.kinetic_hamiltonian + v, 
	    self.vib_basis_size)
	
	return E, V


    def energy_spectrum_alternative(self, psi, el_indices, energy_grid):
	"""
	energy_grid, spectrum = energy_spectrum_alternative(
	    psi, el_indices, energy_grid)
	
	Returns the energy spectrum of the final wavefunction, projecting on 
	the results of the 1/R energy curve.	
	
	Parameters
	----------
	psi : 2D complex array. The wavefunction to be analyzed.
	el_indices : integer list. The indices of the electronic functions for which to 
	    look at the population.  
	energy_grid : 1D float array.
	
	Returns
	-------
	energy_grid : 1D float array.
	spectrum : 1D float array, the energy spectrum of the wavefunction.
	"""
	
	E, V = self.projection_wavefunctions()
	
	spectrum = zeros(len(energy_grid))

	population = zeros(len(E))

	#DEBUG
	population_debug =  zeros(len(E))
	figure()	
	
	for index in el_indices:
	    for j in range(V.shape[1]):
		
		temp_pop = abs(self.inner_product(
		    V[:,j], psi[:,index]))**2
	
#	#TODO test **2 Remove	
#		temp_pop = self.inner_product(
#		    V[:,j], psi[:,index])
#		
		
		
		population[j] += temp_pop
		
		#
		population_debug[j] = temp_pop
#	    plot(E, population_debug)



#
#	#TODO test **2 Remove	
#	population = abs(population)**2

	#Debug
	ion_prob = sum(population)
	#

	#Density of states.
	dos = 1 / diff(E)

	#Choose an appropriate output grid.
	start_index = find(energy_grid > E[0])[0]
	end_index = find(energy_grid < E[-1])[-1] + 1 
	
	spectrum[start_index:end_index] += UnivariateSpline(E[1:], 
	    population[1:] * dos, s=0)(energy_grid[start_index:end_index])	
	
	#debug
	print "ION PROB", ion_prob
	#

	return energy_grid, spectrum


    def energy_spectrum(self, psi, el_indices, energy_grid, energy_offset = None):
	"""
	energy_grid, spectrum = energy_spectrum(psi, el_indices, energy_grid)
	
	Returns the energy spectrum of the final wavefunction.
	
	Parameters
	----------
	psi : 2D complex array. The wavefunction to be analyzed.
	el_indices : integer list. The indices of the electronic functions for which to 
	    look at the population.  
	energy_grid : 1D float array.
	
	Returns
	-------
	energy_grid : 1D float array.
	spectrum : 1D float array, the energy spectrum of the wavefunction.
	"""
	g = tables.openFile(self.name_el_couplings)
	
	try:
	    ref_energies = g.root.E[:,-1] + 1/g.root.R_grid[-1]
	finally:
	    g.close()
	
	f = tables.openFile(self.name_vib_states)

	spectrum = zeros(len(energy_grid))
    
	#Debug
	ion_prob = 0.0
	#

	try:
	    for index in el_indices:
		if energy_offset == None:
		    energies = f.root.E[:,index] + ref_energies[index]#+ - ? TODO
		else:
		    energies = f.root.E[:,index] + energy_offset
		
		population = []

		for j in range(f.root.V.shape[1]):
		    temp_pop = abs(self.inner_product(
			f.root.V[:,j,index], psi[:,index]))**2
		    population.append(temp_pop)
		
		population = array(population)

		#Debug
		ion_prob += sum(population)
		#

		#Density of states.
		dos = 1 / diff(energies)

		#Choose an appropriate output grid.
		start_index = find(energy_grid > energies[0])[0]
		end_index = find(energy_grid < energies[-1])[-1] + 1 

		

		#spectrum[start_index:end_index] += UnivariateSpline(energies, 
		#    population, s=0)(energy_grid[start_index:end_index])	
		spectrum[start_index:end_index] += UnivariateSpline(energies[1:], 
		    population[1:] * dos, s=0)(energy_grid[start_index:end_index])	

	    	
	finally:
	    f.close()
	
	#debug
	print "ION PROB", ion_prob
	#

	return energy_grid, spectrum

    def probability_density(self, psi, el_indices):
	"""
	r_grid, prob = probability_density(psi, el_indices)

	Finds the vibrational probability density of <psi> on the set of electronic states 
	with idices <el_indices>.

	Parameters
	----------
	psi : 2D complex array. The wavefunction.
	el_indices : integer list. The electronic basis functions to be looked at.

	Returns
	-------
	r_grid : 1D float array. The grid on which the probability density is given.
	prob : 1D float array. The probabilty density.
	"""
	prob = zeros(shape(self.grid))
	
	for i in el_indices:
	    coefficients = psi[:,i]
	    
	    if self.splines == None:
		wavefunction_re = self.vib_basis.construct_function_from_bspline_expansion(
		    real(coefficients), self.grid)
		wavefunction_im = self.vib_basis.construct_function_from_bspline_expansion(
		    imag(coefficients), self.grid)
	    else:
		wavefunction_re = dot(self.splines, real(coefficients))
		wavefunction_im = dot(self.splines, imag(coefficients))
	    
	    prob += wavefunction_re**2 + wavefunction_im**2

	return self.grid, prob	
    
    def probability_density_movie(self, el_indices, filename, domain = None):
	"""
	probability_density_movie(el_indices, filename, domain = None)

	Finds the vibrational probability density of psi(t) on the set of electronic states 
	with idices <el_indices>, and makes an animation, <filename>.avi.

	Parameters
	----------
	el_indices : integer list. The electronic basis functions to be looked at.
	filename : string. Name of the animation. (Don't include the extension.)
	domain : float list, on the form [xmin, xmax, ymin, ymax].
	"""
	my_grid = 0
	my_result = []
	
	#Looping over the time steps.
	for i in range(self.psi.shape[2]):
	    my_grid, temp_result = self.probability_density(self.psi[:,:,i], 
		el_indices)
	    my_result.append(temp_result)

	
	#Cast as array.
	my_result = array(my_result)
	
	#Make the animation.
	make_animation.movie_maker(filename, my_grid, my_result,
	    domain = domain)
    
    def make_spline_table(self, my_grid):
	"""
	make_spline_table(my_grid)

	Makes a table of splines on a grid.

	Parameters
	----------
	grid : 1D float array.
	"""
	temp_table = zeros([len(my_grid), self.vib_basis_size])
	for i, grid_point in enumerate(my_grid):
	    for j in range(self.vib_basis_size):
		temp_table[i,j] = self.vib_basis.evaluate_bspline(grid_point,j)
	
	self.splines = temp_table
	self.grid = my_grid



    def fancy_probability_movie(self, el_indices, scaling, filename, domain):
	"""
	fancy_probability_movie(el_indices, scaling, filename, domain)

	Finds the vibrational probability density of psi(t) on the set of electronic states 
	with indices <el_indices>, and makes an animation, <filename>.avi. Fancy version. 

	Parameters
	----------
	el_indices : integer list. The electronic basis functions to be looked at.
	scaling : float list, same length as <el_indices>. 
	    The scaling of the probability.
	filename : string. Name of the animation. (Don't include the extension.)
	domain : float list, on the form [xmin, xmax, ymin, ymax].
	"""

    	if filename not in os.listdir("."):
	    f = tables.openFile(self.name_el_couplings)
	    try:
		E = f.root.E[:]
		R = f.root.R_grid[:]
	    finally:
		f.close()
	    
	    
	    #Get the grid, and thereby the shape.
	    my_grid, temp_result = self.probability_density(self.psi[:,:,0], 
		    [el_indices[0]], domain = domain)
    
	    my_result = zeros([len(my_grid), self.psi.shape[2], len(el_indices)])
	    my_potentials = zeros([len(my_grid), len(el_indices)])
	    
	    #Hurry the plotting along.
	    self.make_spline_table(my_grid)

	    #Looping over the time steps.
	    for j, el_index in enumerate(el_indices):
		my_potentials[:,j] = UnivariateSpline(R,E[j,:] + 1/R,s=0)(my_grid)
		for i in range(self.psi.shape[2]):
		    print i, j, "of", self.psi.shape[2], len(el_indices)  
		    my_grid, temp_result = self.probability_density(
			self.psi[:,:,i], [el_index], domain = domain)
		    my_result[:,i,j] = temp_result.ravel()	
	    #Saving stuff, to avoid doing the heavy lifting again.
	    os.system("mkdir %s"%filename)
	    numpy.save("%s/my_grid"%filename, my_grid)
	    numpy.save("%s/my_result"%filename, my_result)
	    numpy.save("%s/el_indices"%filename, el_indices)
	    numpy.save("%s/my_potentials"%filename, my_potentials)
	    numpy.save("%s/scaling"%filename, scaling)
	else:
	    my_grid =       numpy.load("%s/my_grid.npy"%filename)
	    my_result =     numpy.load("%s/my_result.npy"%filename)
	    my_potentials = numpy.load("%s/my_potentials.npy"%filename)
	    #Can be commented out, to change the scaling.
	    #scaling =       numpy.load("%s/scaling.npy"%filename)
	#Make the animation.
	make_animation.fancy_movie_maker(filename, my_grid, my_result, 
	    my_potentials, scaling, domain)
	

    def energy_spectrum_movie(self, el_indices, energy_grid, filename, domain = None):
	"""
	probability_density_movie(el_indices, energy_grid, filename, domain = None)

	Finds the vibrational probability density of psi(t) on the set of electronic states 
	with idices <el_indices>, and makes an animation, <filename>.avi.

	Parameters
	----------
	el_indices : integer list. The electronic basis functions to be looked at.
	energy_grid : the grid you want the specrum on.
	filename : string. Name of the animation. (Don't include the extension.)
	domain : float list, on the form [xmin, xmax, ymin, ymax].
	"""
	my_result = []
	
	#Looping over the time steps.
	for i in range(self.psi.shape[2]):
	    my_grid, temp_result = self.energy_spectrum(self.psi[:,:,i], 
		el_indices,energy_grid)
	    my_result.append(temp_result)

	
	#Cast as array.
	my_result = array(my_result)
	
	#Make the animation.
	make_animation.movie_maker(filename, my_grid, my_result,
	    domain = domain)

    def inner_product(self, bra, ket):
	"""
	inner_prod = inner_product(bra, ket)
	
	Finds the inner product of two B-spline wavefunctions.

	Parameters
    	----------
	bra : 1D complex array.
	ket : 1D complex array.

	Returns
	-------
	inner_prod : complex.
	"""
	inner_prod = vdot(bra, dot(self.overlap_matrix, ket))
	
	return inner_prod
	

    def full_inner_product(self, bra, ket):
	"""
	full_inner_prod = inner_product(bra, ket)
	
	Finds the inner product of two eigenstate and B-spline wavefunctions.

	Parameters
    	----------
	bra : 1D complex array.
	ket : 1D complex array.

	Returns
	-------
	inner_prod : complex.
	"""
	inner_prod = 0
	for i in range(self.el_basis_size):
	    if len(bra.shape) == 2:
		inner_prod += vdot(bra[:,i], dot(self.overlap_matrix, ket[:,i]))
	    else:
		my_slice = slice(i*self.vib_basis_size, (i+1)*self.vib_basis_size)
		inner_prod += vdot(bra[my_slice], dot(self.overlap_matrix, ket[my_slice]))
	
	return inner_prod


    
