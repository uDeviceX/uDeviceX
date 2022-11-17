.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp
.SUFFIXES: .o

NVCC = nvcc
MPICXX = mpicxx.mpich
HDF5_LDFLAGS = ../lib/libudx.a,/usr/lib/x86_64-linux-gnu/hdf5/mpich/libhdf5.a,-lsz,-lz
CXXFLAGS = -Ofast

all: main
main: main.o
.cpp.o:; $(MPICXX) -c $(CXXFLAGS) $< -I../lib
main: main.o; $(NVCC) -lcurand -ccbin=$(MPICXX) --linker-options=$(HDF5_LDFLAGS) $< -o $@

clean:; rm -f main main.o
