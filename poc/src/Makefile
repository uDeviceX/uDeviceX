.POSIX:
.SUFFIXES:
.SUFFIXES: .o
.SUFFIXES: .c
.SUFFIXES: .cpp
.SUFFIXES: .cu

MPICXX = mpicxx
MPICC = mpicc
HDF5_CXXFLAGS = `pkg-config --cflags hdf5-mpich`
NVCC = nvcc
NVCCFLAGS = -arch=sm_35
CXXFLAGS = -Ofast
CFLAGS = -Ofast

include var.mk
all: libudx_c.a libudx_g.a
.cpp.o:
	$(MPICXX) -c -I. -I`dirname $<` $(CXXFLAGS) $(HDF5_CXXFLAGS) $< -o $@
.c.o:
	$(MPICC) -c -I. -I`dirname $<` $(CFLAGS) $< -o $@
.cu.o:
	$(NVCC) $(NVCCFLAGS) -c -I. -I`dirname $<` $(CXXFLAGS) $(HDF5_CXXFLAGS) $< -o $@

libudx_c.a: $(O_C); ar r $@ $(O_C) && ranlib $@
libudx_g.a: $(O_G); ar r $@ $(O_G) && ranlib $@


clean:; rm -rf libudx_g.a libudx_c.a $(O_C) $(O_G)
