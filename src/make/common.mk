# common part of user configs
GSL_DIR      = `gsl-config --prefix`
GSL_CXXFLAGS = `gsl-config --cflags`
GSL_LIBS     = `gsl-config --libs` -L${GSL_DIR}/lib -Wl,-rpath -Wl,${GSL_DIR}/lib

NVCCFLAGS = --compiler-options \
	 '${CXXFLAGS}          \
	  ${HDF5_CXXFLAGS} ${MPI_CXXFLAGS} ${GSL_CXXFLAGS}'
LIBS =    ${HDF5_LIBS}  ${MPI_LIBS}  ${GSL_LIBS} ${NVCC_LIBS}

NVCCLIBS  = --linker-options '${HDF5_LIBS}     ${MPI_LIBS}     ${GSL_LIBS}'
