"""
This program assumes that mencoder (http://en.wikipedia.org/wiki/MEncoder, 2012)
is installed on the machine.
"""
from pylab import figure

import os

def plotter(my_grid, my_result, domain = None):
    """
    plotter(my_grid, my_result, domain = None)

    Plots the functions, and stores the image as a png file.

    Parameters
    ----------
    my_grid : 1D float array. 
    my_result : 2D float array. Function on the grid for different times.
    domain : float list, length 4, in this way: [xmin, xmax, ymin, ymax].
	Will be constructed if it is not supplied.
    """
    if domain == None:
	#Find the logical axes of the figures.
	domain = [my_grid[0], my_grid[-1], 
	    min(my_result.ravel()), max(my_result.ravel())]
    
    else:
	domain.append( min(my_result.ravel()))
	domain.append( max(my_result.ravel()))
    
    #Open figure. 
    fig = figure()
    
    #Find out if time is on 0 or 1 axis in my_result.
    time_axis = 1 - (my_result.shape[1] == my_grid.shape[0])
    
    #Loop over times.
    for i in range(my_result.shape[time_axis]):
	#Set aspect and domain/range.
	ax = fig.add_subplot(111)
	ax.set_xlim((domain[0], domain[1]))
	ax.set_ylim((domain[2], domain[3]))
	
	#Handles arbitrary choice of time axis in my_result.
	if time_axis == 0:
	    ax.plot(my_grid, my_result[i,:])
	else:
	    ax.plot(my_grid, my_result[i,:])
	
	#Stores image as a png.
	fig.savefig("frames_%03i.png"%(i), dpi = 100)
	
	#Clear the figure.
	fig.clf()

def movie_maker(filename, my_grid, my_result, domain = None):
    """
    movie_maker(filename, my_grid, my_result, domain = None)

    Makes an animation of <my_results> on <my_grid>, and stores it in 
    <filename>.avi.

    Parameters
    ----------
    filename : string. Name of the movie file. (Extension 
	added automatically.)
    my_grid : 1D float array. 
    my_result : 2D float array. Function on the grid for different times.
    domain : float list, length 4, in this way: [xmin, xmax, ymin, ymax].
	Will be constructed if it is not supplied.
    """
    
    #Makes the png files.
    plotter(my_grid, my_result, domain = domain)

    #Make the movie.
    os.system("bash mencoder.sh")

    #Copy the outpyt file, and remove the temporary directory.
    os.system("cp output.avi %s.avi"%(filename))
    os.system("rm frame*.png")
    os.system("rm output.avi")


def fancy_plotter(my_grid, my_result, my_potentials, scaling, domain):
    """
    fancy_plotter(my_grid, my_result, my_potentials, scaling, domain)

    Plots the functions, and stores the image as a png file.

    Parameters
    ----------
    my_grid : 1D float array. 
    my_result : 2D float array. Function on the grid for different times.
    my_potentials : 2D float array. Function on the grid for different potentials.
    scaling : float list, the scaling on each el_index.
    domain : float list, length 4, in this way: [xmin, xmax, ymin, ymax].
    """
    
    #Open figure. 
    fig = figure()
    
    #Loop over times.
    for i in range(my_result.shape[1]):
	#Set aspect and domain/range.
	ax = fig.add_subplot(111)
	ax.set_xlim((domain[0], domain[1]))
	ax.set_ylim((domain[2], domain[3]))
	for j in range(my_result.shape[2]):
	    ax.plot(my_grid, scaling[j] * my_result[:,i,j] + my_potentials[:,j])
	    ax.plot(my_grid, my_potentials[:,j],'k')
	
	#Stores image as a png.
	fig.savefig("frames_%03i.png"%(i), dpi = 100)
	
	#Clear the figure.
	fig.clf()

def fancy_movie_maker(filename, my_grid, my_result, 
	    my_potentials, scaling, domain):
    """
    movie_maker(filename, my_grid, my_result, domain)

    Makes an animation of <my_results> on <my_grid>, and stores it in 
    <filename>.avi.

    Parameters
    ----------
    filename : string. Name of the movie file. (Extension 
	added automatically.)
    my_grid : 1D float array. 
    my_result : 2D float array. Function on the grid for different times.
    domain : float list, length 4, in this way: [xmin, xmax, ymin, ymax].
    """
    
    #Makes the png files.
    fancy_plotter(my_grid, my_result, my_potentials, scaling, domain)

    #Make the movie.
    os.system("bash mencoder.sh")

    #Copy the outpyt file, and remove the temporary directory.
    os.system("cp output.avi %s.avi"%(filename))
    os.system("rm frame*.png")
    os.system("rm output.avi")






    
