NVCCFLAGS = --compiler-options \
	 '${CXXFLAGS}          \
	  ${HDF5_CXXFLAGS} ${MPI_CXXFLAGS}'
LIBS =    ${HDF5_LIBS}  ${MPI_LIBS}  ${NVCC_LIBS}

NVCCLIBS  = --linker-options '${HDF5_LIBS}     ${MPI_LIBS}'
