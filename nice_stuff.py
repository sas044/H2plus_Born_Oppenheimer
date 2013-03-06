



def distribute_work(nr_proc, nr_tasks, my_id):
    """
    my_tasks = distribute_work(nr_proc, nr_tasks, my_id)
    
    When distributing <nr_tasks> tasks among <nr_proc> processors, 
    what tasks belong to processor <my_id>?

    Parametres
    ----------
    nr_proc : integer, the number of slave processors.
    nr_tasks : integer, the number of tasks.
    my_id : integer, the 'name' of the processor asking.

    Returns
    -------
    my_tasks : integer list, containing the indices of the tasks 
	that are processor <my_id>'s share. 

    Notes
    -----
    This program assumes that the processor numbers, my_id, start from 0.
    """
    #Distributing the simple stuff.
    main_workload = nr_tasks/nr_proc
    remainder = nr_tasks%nr_proc
    my_tasks = range(my_id * main_workload, (my_id + 1) * main_workload)
    
    #Distributing the remainder.
    if my_id < remainder :
	my_tasks.append(nr_tasks - my_id - 1)

    return my_tasks



#Progress visualization
def status_bar(comment, current_index, total, first = 0, graphic = True):
    """
    status_bar(comment, current_index, total, first = 0, graphic=True)


    Prints a status bar on your console, showing how far a for loop has run.
    Used in a for loop. No other printing should go on in the for loop. 


    Parametres
    ----------
    comment : string, containing the title of the status bar. 
    
    current_index : integer, the current index in the for loop.
    
    total : integer, the end index of the loop, i.e. 
	    for i in range(total):
    
    first : integer, the starting index, the default value being 0.
    
    graphic : boolean. If True (default), there will be printed a status bar,
	    if False, the format will be:
	    <comment> <current_index> of <total>
	    

    Examples
    --------
    >>>import time
    >>>for i in range(1000):
	    status_bar("Testing the status bar:", i, 1000)
	    time.sleep(.01)


    Testing the status bar: [=========================                   ]
    """    
    import sys
    

    if(graphic):
	total_length = 50
	if(current_index == first):
	    sys.stdout.flush()
	    my_string = make_string(current_index,total,total_length,graphic)
	    status_string = "\n%s\t%s"%(comment, my_string)
	    sys.stdout.write(status_string)
	else:
	    sys.stdout.flush()
	    my_string = make_string(current_index,total,total_length,graphic)
	    for i in range(2+total_length):
		sys.stdout.write("\b")
	    sys.stdout.write(my_string)

    else:
	total_length = len('%i'%total)

	if(current_index == first):
	    sys.stdout.flush()
	    my_string = make_string(current_index,total,total_length,graphic)
	    status_string = "%s\t%s"%(comment, my_string)
	    sys.stdout.write(status_string)
	else:
	    sys.stdout.flush()
	    my_string = make_string(current_index,total,total_length,graphic)
	    for i in range(2+total_length):
		sys.stdout.write("\b\b")
	    sys.stdout.write(my_string)


def make_string(current_index, total,total_length, graphic):
    """
    makes the correct string:
    """
    if(graphic):
	my_string = "["
	for i in range(total_length):
	    if(i > total_length*current_index/total):
		my_string += " "
	    else:
		my_string += "="
	my_string += "]"

    else:
	tmp = "        "
	tmp = tmp.join([tmp,str(current_index+1)])
	current = tmp[-total_length:]
	my_string = "%s of %i"%(current,total)
    return my_string

