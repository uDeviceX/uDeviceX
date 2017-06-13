# system and user configuration

## compiler

* `NVCC_BIN`   path
* `CXXFLAGS` `*_CXXFLAGS`

## linker

* `LINK`       path
* `*_LDFLAGS`
* `*_LIBS`

## mpi

To get the flags

	mpicxx -show
or

	pkg-config --libs mpich

## hdf5 library

To get flags

	h5c++ -show

or

	pkg-config hdf5-mpich --libs

To build from source

https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.17/src/hdf5-1.8.17.tar.gz

Configuration options

      --prefix=$HOME/prefix/hdf5 
	  --enable-parallel
	  CXX=/usr/lib64/mpich/bin/mpic++
	  CC=/usr/lib64/mpich/bin/mpicc

## gsl

## nvcc

How did we get flags?
