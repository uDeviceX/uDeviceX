.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp

MPICXX = mpicxx.mpich
HDF5_LDFLAGS = `pkg-config --libs hdf5-mpich`
CXXFLAGS = -Ofast

all: main
.cpp:
	$(MPICXX) $(CXXFLAGS) $< -I../lib ../lib/libudx.a -lcurand -lcudart $(HDF5_LDFLAGS) -o $@
