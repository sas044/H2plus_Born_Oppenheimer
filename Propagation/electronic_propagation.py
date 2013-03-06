from scipy.integrate import odeint
from pylab import shape, find
from numpy import zeros, pi, sin, cos, dot, r_, diff, mean
from numpy import real, imag,  diag, linspace, argmin
from scipy.interpolate import UnivariateSpline

import tables

"""
Propagate the electronic system, using the fixed nuclei approximation.
"""

def propagate(R):
    """
    y_out = propagate()
    Runs the electronic propagation.
    """

    #Laser parameters.
    amplitude = 0.17 #I = 1e14 #0.01688#I = 1e13
    omega = 1.2
    cycles = 20
    extra_time = 0.00
    r_value = R

    #filename = "../Data/el_couplings_lenght_z_m_0_nu_71_mu_31_beta_1_00_theta_0_00_m_0_q_2_E_0_-20.h5"
    filename = "../Data/el_couplings_lenght_z_m_0_nu_71_mu_31_beta_1_00_theta_0_00_m_0_q_4_E_5_00.h5"
    input_function = Ode_function_call(r_value, filename, amplitude, omega, cycles)

    #Initial state.
    y_0 = zeros([2 * input_function.basis_size])
    y_0[0] = 1.0
    
    #OBS! DEBUG! 2
    times = linspace(0, 2 * 2. * pi * cycles/omega + extra_time, 300)
    
    #Solving the TDSE.
    y, information = odeint(input_function.dpsi_dt, y_0, times, printmessg=1, full_output = 1)#, hmax = 0.03)

    y_out = zeros([y.shape[0], y.shape[1]/2], dtype = complex)
    y_out = y[:, :y.shape[1]/2] + 1j * y[:, y.shape[1]/2:] 

    return y_out, information, input_function


class Electronic_analysis:
    """
    This class provides analysis tools for an electronic wavefunction.
    """

    def __init__(self, psi, information, input_function):
	"""
	Electronic_analysis(y_out, information, input_function)

	Constructor.

	Parameters
	----------
	psi : 2D real array. The raw version of the wavefunction, 
	    for all times.
	information : the stuff odeint() spews out concerning the propagation.
	input_function : Ode_function_call instance, used in the propagation.
	"""

	self.psi = psi
	self.psi_final = psi[-1,:]
	self.p = real(abs(self.psi)**2)
	self.info = information
	self.input_function = input_function
	
	self.make_continuum_list()

    def norm(self, psi):
	"""
	n = norm(psi)

	Calculates the norm: int( |psi|**2 )True

	Parameters
	----------
	psi : 1D complex array. Some wavefunction.

	Returns
	-------
	n : real, the integral of the absolute square of the wavefunction.
	"""

	n = sum(abs(psi)**2)
	return n

    def make_continuum_list(self):
	"""
	Creates a list of the states that are in the continuum, 
	based on the density of states. 
	"""
	energies = diag(self.input_function.H_0)
	self.energies = energies
	continuum = []
	border_value_list = []

	#Looping the q's.
	for i in range(max(self.input_function.index_array[:,1])+1):

	    #Get the indices corresponding to  given q.
	    temp_ind = find(self.input_function.index_array[:,1] == i)
	    
	    #Corresponding energies.
	    temp_E = energies[temp_ind]
	    
	    #The continuum starts when gap between energies increases.
	    border_value = temp_ind[argmin(diff(temp_E))]
	    border_value_list.append(energies[border_value])

	    continuum.append(temp_ind[find((temp_ind > border_value))])
	
	self.continuum = [item for sublist in continuum for item in sublist]
	self.continuum_limit_dos = mean(border_value_list)
    
    def ionization_probability(self, psi):
	"""
	ion_prob = ionization_probability(psi)

	Calculates the ionization probability.

	Parameters
	----------
	psi : 1D complex array, the wavefunction in question.

	Returns
	-------
	ion_prob : real, the probability of ionization, based on DOS.
	"""
	ion_prob = sum(abs(psi[self.continuum])**2)

	return ion_prob

    def energy_spectrum(self, psi, energy_grid):
	"""
	energy_grid, spectrum = energy_spectrum(psi, energy_grid)

	Creates the energy spectrum on the <energy_grid>.

	"""
	all_energies = diag(self.input_function.H_0)
	all_population = abs(psi)**2

	spectrum = zeros(len(energy_grid))
    
	#Density of states.
	for q in range(max(self.input_function.index_array[:,1])+1):
	    q_indices = find(self.input_function.index_array[:,1] == q)
	    energies = all_energies[q_indices]
	    population = all_population[q_indices]
	    dos = 1 / diff(energies)
	    if True:
		#Choose an appropriate output grid.
		start_index = find(energy_grid > energies[0])[0]
		end_index = find(energy_grid < energies[-1])[-1] + 1 

		spectrum[start_index:end_index] += UnivariateSpline(energies[1:], 
		    population[1:] * dos, s=0)(energy_grid[start_index:end_index])	
	    else:
		spectrum += UnivariateSpline(energies[1:], 
		    population[1:] * dos, s=0)(energy_grid)
	    #raise
	return energy_grid, spectrum

#	q = 1
#	q_indices = find(self.input_function.index_array[:,1] == q)
#	energies = all_energies[q_indices]
#	population = all_population[q_indices]
#	dos = 1 / diff(energies)
#	
#	return energies[1:], population[1:] * dos
#
#

class Ode_function_call:
    """
    This class provides a function that can serve as input to 
    scipy.integrate.odeint().
    """
    
    def __init__(self, r_value, filename_coupling, amplitude, omega, cycles):
	"""
	Ode_function_call()

	Constructor.

	Parameters
	----------
	filename_coupling : string, path to the HDF5 file that contains the 
	    dipole couplings of the electronic H2+ problem.
	"""
	#Laser info.
	self.amplitude = amplitude
	self.omega = omega
	self.cycles = cycles
	self.pulse_duration = 2* pi /(self.omega) * self.cycles

	#Open files
	f = tables.openFile(filename_coupling)

	#Retrieve r value.
	r_grid = f.root.R_grid[:]
	self.r_index = argmin(abs(r_grid - r_value))	
	self.index_array = f.root.index_array[:]
	self.r = r_grid[self.r_index]

	#Retrieve Hamiltonian.
	self.H_0 = diag(f.root.E[:,self.r_index])
	self.H_1 = f.root.couplings[:,:,self.r_index]	
	
	#Close file.
	f.close()

	#Basis sizes.
	self.basis_size = shape(self.H_0)[0]

    def dpsi_dt(self, psi, t):
	"""
	dp_dt = dpsi_dt(psi, t)

	Method that serves as input to the odeint() function.
	Calculates dpsi/dt = -i S**-1 H(t) psi.

	Parameters
	----------
	psi : 1D complex array. Wavefunction.
	t : float. Time.

	Returns
	-------
	dp_dt : 1D complex array. Derivative of the wavefunction.
	"""
	#Making a complex array. 
	psi_complex = psi[:len(psi)/2] + 1j * psi[len(psi)/2:] 
	
	#Do operations.
	dp_dt_complex = self.mat_vec_product(psi_complex, t)

	#Making a float array.
	dp_dt = r_[real(dp_dt_complex), imag(dp_dt_complex)] 
	
	return dp_dt

    def mat_vec_product(self, psi, t):
	"""
	psi_final = mat_vec_product(psi, t)

	Does the matrix vector multiplication with the Hamiltonian.

	Parameters
	----------
	psi : 1D complex array. Wavefunction.
	t : float. Time.

	Returns
	-------
	psi_final : 1D complex array. Result of the multiplication.

	Notes
	-----
	In the present form, one assumes real couplings, i.e. symmetric H.
	"""
	
	#Time independent part.
	x = zeros(self.basis_size, dtype=complex)
	#Time dependent part.
	y = zeros(self.basis_size, dtype=complex)
	

	#Multiply with the time independent hamiltonian.
	x = dot(self.H_0, psi)
	y = dot(self.H_1, psi)
		
	#Weigh with field strength, and add components.
	psi_final =  x + self.time_function(t) * y
	
	return -1j * psi_final


    def time_function(self, t):
	"""
	field_strength = time_function(t)

	Returns the electrical field strength at the time <t>.
	Assumes an electrical field on the form 
	E(t) = sin**2(pi * t / T) * cos(omega * t)

	Parameters
	----------
	t : float. A time.

	Returns
	-------
	field_strength : float. Electrical field strength at the time <t>.
	"""
	#OBS! DEBUG
	if t > 2 * self.pulse_duration:
	    field_strength = 0.0
	else:
	    field_strength = (self.amplitude * 
		sin(pi * t / self.pulse_duration)**2 *  
		cos(self.omega * t))
	
	return field_strength


