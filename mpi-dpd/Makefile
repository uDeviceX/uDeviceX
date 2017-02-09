-include .cache.Makefile

NVCC ?= nvcc -ccbin $(CXX)
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35

NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL) -O3 -use_fast_math -g -std=c++11
CXXFLAGS  += -L../cuda-dpd/dpd -L../cuda-rbc/    -O3 -g  -std=c++11 -DNDEBUG
NVCCFLAGS += -I../cuda-dpd/dpd -I../cuda-rbc/

OBJS = \
containers.o wall.o dpd.o \
fsi.o contact.o redistribute-particles.o \
redistribute-rbcs.o solvent-exchange.o solute-exchange.o \
common.o io.o scan.o \
minmax.o redistancing.o last_bit_float.o \
visc-aux.o simulation.o main.o \
velcontroller.o

LIBS += -lcuda-dpd -lcuda-rbc -lcudart

ifeq ($(h5part),0)
NVCCFLAGS += -DNO_H5PART
endif

ifeq ($(h5),0)
NVCCFLAGS += -DNO_H5
endif

slevel ?= 0
datadump ?= 1

ifeq ($(datadump),0)
NVCCFLAGS += -D_NO_DUMPS_
endif
ifeq ($(datadump),1)
NVCCFLAGS += -D_SYNC_DUMPS_
endif

NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel)

# I have some device code in OBJS. I have to link it separatly
test:  $(OBJS) libcuda-dpd libcuda-rbc
	$(NVCC) $(NVCCFLAGS) -dlink -L../cuda-dpd/dpd $(OBJS) $(LIBS) -o gpuCode.o
	$(CXX)  $(CXXFLAGS) gpuCode.o $(OBJS) $(LIBS) -o test

# use `-dc' flag to do separate compilation
# Separate Compilation and Linking of CUDA C++ Device Code
# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
containers.o: containers.cu
	$(NVCC) $(NVCCFLAGS) -dc $<            -c -o $@

wall.o: wall.cu
	$(NVCC) $(NVCCFLAGS) -dc $<            -c -o $@

fsi.o: fsi.cu
	$(NVCC) $(NVCCFLAGS) -dc $<            -c -o $@

contact.o: contact.cu
	$(NVCC) $(NVCCFLAGS) -dc $<            -c -o $@

last_bit_float.o: last_bit_float.cu
	$(NVCC) $(NVCCFLAGS) -dc $<            -c -o $@

dpd.o: dpd.cu
	$(NVCC) $(NVCCFLAGS) -dc $<            -c -o $@

visc-aux.o: visc-aux.cu
	$(NVCC) $(NVCCFLAGS) -dc $<            -c -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

libcuda-dpd:
	make -C ../cuda-dpd/dpd libcuda-dpd.a \
	CXX="$(CXX)" NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)"

libcuda-rbc:
	make -C ../cuda-rbc libcuda-rbc.a \
	CXX="$(CXX)" NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)"

clean:
	rm -f test *.o

cleanall: clean
	make -C ../cuda-dpd/dpd clean
	make -C ../cuda-rbc clean

.PHONY = clean libcuda-dpd libcuda-rbc
