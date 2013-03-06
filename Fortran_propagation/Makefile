###
###HEX
###
FC = ftn# gfortran
FLAGS = -O3 -ffree-form -ffree-line-length-none
LIBFLAGS =  -lgfortran #-llapack 

HDF5MAINPATH = /opt/cray/hdf5/1.8.8/gnu/47
HDF5FLAGS_1 = -I$(HDF5MAINPATH)/include
HDF5FLAGS_2 = $(HDF5FLAGS_1) -L$(HDF5MAINPATH)/lib $(HDF5MAINPATH)/lib/libhdf5hl_fortran.a $(HDF5MAINPATH)/lib/libhdf5_hl.a $(HDF5MAINPATH)/lib/libhdf5_fortran.a $(HDF5MAINPATH)/lib/libhdf5.a -lpthread  -lm -Wl,-rpath -Wl,/usr/lib -Wl,-Bsymbolic-functions

default: main

ode.o: ode.f90
	$(FC) $(FLAGS) -fPIC -c ode.f90  

propagator.o: propagator.f90 
	$(FC) $(FLAGS) $(LIBFLAGS) $(HDF5FLAGS_1) -c propagator.f90 

main: propagator.o main.f90 ode.o
	$(FC)  main.f90 propagator.o ode.o -o main $(FLAGS) $(LIBFLAGS) $(HDF5FLAGS_2)


clean: 
	rm propagator.o propagator.mod ode.o


###
###SKULE
###
#FC = mpif90#gfortran
#FLAGS = -O3 -ffree-form -ffree-line-length-none
#LIBFLAGS =  -lgfortran -llapack -L/usr/lib/
#
#HDF5MAINPATH = /usr/local/hdf5
#HDF5FLAGS_1 = -I$(HDF5MAINPATH)/include
#HDF5FLAGS_2 = $(HDF5FLAGS_1) -L$(HDF5MAINPATH)/lib $(HDF5MAINPATH)/lib/libhdf5hl_fortran.a $(HDF5MAINPATH)/lib/libhdf5_hl.a $(HDF5MAINPATH)/lib/libhdf5_fortran.a $(HDF5MAINPATH)/lib/libhdf5.a -lpthread  -lm -Wl,-rpath -Wl,/usr/lib -Wl,-Bsymbolic-functions
#
##MPIPATH = /usr/lib/openmpi/
##MPIFLAGS = -L$(MPIPATH)lib -I$(MPIPATH)include -l$(MPIPATH)mpi_f90
#
#
#
#default: main
#
#ode.o: ode.f90
#	$(FC) $(FLAGS) -fPIC -c ode.f90  
#
#propagator.o: propagator.f90 
#	$(FC) $(FLAGS) $(LIBFLAGS) $(HDF5FLAGS_1) -c propagator.f90 
#
#main: propagator.o main.f90 ode.o 
#	$(FC)  main.f90 propagator.o ode.o  -o main $(FLAGS) $(LIBFLAGS) $(HDF5FLAGS_2)
#
#
#clean: 
#	rm propagator.o propagator.mod ode.o
#

