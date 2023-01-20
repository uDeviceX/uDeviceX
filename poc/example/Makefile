.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp
.SUFFIXES: .o

NVCC = nvcc
MPICXX = mpicxx
HDF5_LDFLAGS = ../lib/libudx.a,/usr/lib/x86_64-linux-gnu/hdf5/mpich/libhdf5.a,-lsz,-lz
CXXFLAGS = -Ofast

all: main hdf
main: main.o
hdf: hdf.o
.cpp.o:; $(MPICXX) -c $(CXXFLAGS) $< -I../lib
main: main.o; $(NVCC) -lcurand -ccbin=$(MPICXX) --linker-options=$(HDF5_LDFLAGS) $< -o $@
hdf: hdf.o; $(NVCC) -lcurand -ccbin=$(MPICXX) --linker-options=$(HDF5_LDFLAGS) $< -o $@

clean:; rm -f main main.o hdf5 hdf5.o
