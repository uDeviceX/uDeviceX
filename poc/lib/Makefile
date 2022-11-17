.POSIX:
.SUFFIXES:
.SUFFIXES: .o
.SUFFIXES: .c
.SUFFIXES: .cpp
.SUFFIXES: .cu

MPICXX = mpicxx.mpich
MPICC = mpicc.mpich
HDF5_CXXFLAGS = `pkg-config --cflags hdf5-mpich`
NVCC = nvcc
NVCCFLAGS = -arch=sm_35
CXXFLAGS = -Ofast
CFLAGS = -Ofast

include var.mk
libudx.a: $O; ar r $@ $O && ranlib $@
.cpp.o:
	$(MPICXX) -c -I. -I`dirname $<` $(CXXFLAGS) $(HDF5_CXXFLAGS) $< -o $@
.c.o:
	$(MPICC) -c -I. -I`dirname $<` $(CFLAGS) $< -o $@
.cu.o:
	$(NVCC) -c -I. -I`dirname $<` $(NVCCFLAGS) $(HDF5_CXXFLAGS) $< -o $@


clean:; rm -f libudx.a $O
