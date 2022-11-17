.POSIX:
.SUFFIXES:
.SUFFIXES: .c
.SUFFIXES: .cpp
.SUFFIXES: .cu
.SUFFIXES: .o

MPICXX = mpicxx.mpich
CC = c99
HDF5_CXXFLAGS = `pkg-config --cflags hdf5-mpich`
NVCC = nvcc
NVCCFLAGS = -arch=sm_75
CXXFLAGS = -Ofast
CFLAGS = -Ofast

include var.mk
libudx.a: $O; ar r $@ $O && ranlib $@
.c.o:; $(CC) -c $(CFLAGS) $< -o $@
.cpp.o:; $(MPICXX) -c -I`dirname $<` -I. $(CXXFLAGS) $(HDF5_CXXFLAGS) $< -o $@
.cu.o:; $(NVCC) -ccbin=$(MPICXX) -c -I`dirname $<` -I. $(NVCCFLAGS) $< -o $@
clean:; rm -f libudx.a $O
