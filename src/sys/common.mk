GSL_DIR      = `gsl-config --prefix`
GSL_CXXFLAGS = `gsl-config --cflags`
GSL_LDFLAGS  = -L${GSL_DIR}/lib -Wl,-rpath -Wl,${GSL_DIR}/lib
GSL_LIBS     = `gsl-config --libs`

NVCC = ${NVCC_BIN} --compiler-options \
	  '${HDF5_CXXFLAGS} ${MPI_CXXFLAGS} ${GSL_CXXFLAGS}'
LDFLAGS = ${HDF5_LDFLAGS}  ${MPI_LDFLAGS}  ${GSL_LDFLAGS} ${NVCC_LDFALGS}
LIBS =    ${HDF5_LIBS}     ${MPI_LIBS}     ${GSL_LIBS}
