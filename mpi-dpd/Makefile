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

inquire: $(bash [ `cat slevel.txt` == "$(slevel)" ] || { echo "cleanall" ; } )
	$(bash echo "$(slevel)" > slevel.txt )
	$(MAKE) $(MFLAGS) test

# I have some device code in OBJS. I have to link it separatly
test:  $(OBJS) libcuda-dpd libcuda-rbc
	echo h5part is "$(h5part)"
	$(NVCC) $(NVCCFLAGS) -dlink -L../cuda-dpd/dpd $(OBJS) $(LIBS) -o gpuCode.o
	$(CXX)  $(CXXFLAGS) gpuCode.o $(OBJS) $(LIBS) -o test

-include $(OBJS:.o=.d)

# use `-dc' flag to do separate compilation
# Separate Compilation and Linking of CUDA C++ Device Code
# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
containers.o: containers.cu
	$(NVCC) $(NVCCFLAGS) -dc $*.cu            -c -o $*.o
	$(NVCC) $(NVCCFLAGS) -M  $*.cu           >      $*.d
wall.o: wall.cu
	$(NVCC) $(NVCCFLAGS) -dc $*.cu            -c -o $*.o
	$(NVCC) $(NVCCFLAGS) -M  $*.cu           >      $*.d
fsi.o: fsi.cu
	$(NVCC) $(NVCCFLAGS) -dc $*.cu            -c -o $*.o
	$(NVCC) $(NVCCFLAGS) -M  $*.cu           >      $*.d
contact.o: contact.cu
	$(NVCC) $(NVCCFLAGS) -dc $*.cu            -c -o $*.o
	$(NVCC) $(NVCCFLAGS) -M  $*.cu           >      $*.d
last_bit_float.o: last_bit_float.cu
	$(NVCC) $(NVCCFLAGS) -dc $*.cu            -c -o $*.o
	$(NVCC) $(NVCCFLAGS) -M  $*.cu           >      $*.d
dpd.o: dpd.cu
	$(NVCC) $(NVCCFLAGS) -dc $*.cu            -c -o $*.o
	$(NVCC) $(NVCCFLAGS) -M  $*.cu           >      $*.d
visc-aux.o: visc-aux.cu
	$(NVCC) $(NVCCFLAGS) -dc $*.cu            -c -o $*.o
	$(NVCC) $(NVCCFLAGS) -M  $*.cu           >      $*.d

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $*.cu -c -o $*.o
	$(NVCC) $(NVCCFLAGS) -M $*.cu > $*.d

libcuda-dpd:
	$(MAKE) -C ../cuda-dpd/dpd libcuda-dpd.a CXX="$(CXX)" NVCC="$(NVCC)" ARCH_VAL="$(ARCH_VAL)" CODE_VAL="$(CODE_VAL)" slevel="$(slevel)"

libcuda-rbc:
	$(MAKE) -C ../cuda-rbc/ libcuda-rbc.a CXX="$(CXX)" NVCC="$(NVCC)" ARCH_VAL="$(ARCH_VAL)" CODE_VAL="$(CODE_VAL)"

clean:
	rm -f test *.o *.d slevel.txt

cleanall: clean
	rm -f slevel.txt
	make -C ../cuda-dpd/dpd clean
	make -C ../cuda-rbc clean

#we want to give always a try to update the libs - too complicated to track dependecies there
.PHONY = clean libcuda-dpd libcuda-rbc inquire
