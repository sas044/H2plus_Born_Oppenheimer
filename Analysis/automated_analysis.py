"""
For convenience, a program to do multiple analyses. Somewhat poorly documented. Probably safer to stick with analysis.py.
"""

import sys
sys.path.append("../Fortran_propagation/")

import os
import basis_truncation as trunc
import analysis

def analysis_operation(A, conf):
    """
    grid, dP = analysis_operation(A, conf)

    Given a 'loaded' analysis object, this method retrieves the analyzed 
    results.

    Parameters
    ----------
    A : Analysis object, with loaded wavefunction.
    conf : truncated basis config object.

    Returns
    -------
    grid : 1d float array, output domain.
    dP : 1d float array, output data.
    """
    ind = 4 #should be calculated from index in original el_coupling.
    grid, dP = A.energy_spectrum(A.psi_final, [ind], r_[0:1:.005],energy_offset = 0.5)
    
    return grid, dP

def analyze_one(name_out):
    """
    grid, dP = analyze_one(name_out)

    Given a result path <name_out>, this program searches for the result on 
    the local machine, and on Hex. It then does the appropriate analysis, 
    according to analysis_operation(), and returns the result.

    Parameters
    ----------
    name_out : string, path to a propagated wavefunction. 
	Should at least contain folder and filename, 
	and then it should be able to fix the rest.

    Returns
    -------
    grid : 1d float array, output domain.
    dP : 1d float array, output data.
    """
    #Checks and retrieval of propagation file.
    #-----------------------------------------

    #Clean up name_out.
    hex_path_string = ("sas044@hexagon.bccs.uib.no:" + 
	"/work/sas044/BO_H2plus/Fortran_propagation/output/")
    path_string = "../Propagation/output/"
    dir_name = name_out.split("/")[-2]
    file_name = name_out.split("/")[-1]
    

    #Check local machine.

    #Check for directory, and if necessary, retrieve from Hex.
    if name_out.split("/")[-2] not in os.listdir(path_string):
	os.system("scp -r %s%s %s"%(
	    hex_path_string, dir_name, path_string))
    
    conf = trunc.Truncation_config(config = " %s/%s/info.txt"%(
	path_string, dir_name))

    #Check for file, and if necessary, retrieve from Hex.
    if file_name not in os.listdir("%s%s"%(path_string, dir_name)):
	try:
	    #Temporary directory for the retrieved file and info.txt.
	    os.system("mkdir %s/%s/temp"%(path_string, dir_name))
	    os.system("scp %s%s/%s %s/%s/temp/"%(hex_path_string, 
		dir_name, file_name, path_string, dir_name))
	    os.system("scp %s%s/info.txt %s/%s/temp/"%(hex_path_string, 
		dir_name, path_string, dir_name))
	    
	    #Compare the truncation config objects.
	    conf_temp = trunc.Truncation_config(
		config = " %s/%s/temp/info.txt"%(path_string, dir_name))
	    
	    assert conf.equals(conf_temp)
	    
	    os.system("mv %s/%s/temp/%s %s/%s/"%(path_string, dir_name, 
		file_name, path_string, dir_name,))

	finally:
	    #os.system("rm -r %s/%s/temp/"%(path_string, dir_name))


    #Analysis
    #--------
    A = analysis.Analysis("%s%s"%(path_string, dir_name))
    if "%snpy"%file_name[:-3] in os.listdir("%s%s"%(path_string, dir_name)):
	A.retrieve_result("%snpy"%file_name[:-3])
    else:
	A.retrieve_result(file_name)
    
    grid, dP = analysis_operation(A, conf)

    return grid, dP
	
    
