include .cache.Makefile # local settings (see .cache.Makefile.* for examples)

NVCC     ?= nvcc -ccbin $(CXX)
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35
OPT ?= -O3 -use_fast_math -g
# for cuda-gdb
#OPT      ?= -O0 -g --device-debug

slevel   ?= 0
datadump ?= 1

NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL) -std=c++11
LDFLAGS   += -L../cuda-dpd/dpd -L../cuda-rbc
NVCCFLAGS += -I../cuda-dpd/dpd -I../cuda-rbc
LIBS      += -lcuda-dpd -lcuda-rbc -lcudart

OBJS = \
containers.o wall.o dpd.o \
fsi.o contact.o redistribute-particles.o \
redistribute-rbcs.o solvent-exchange.o solute-exchange.o \
common.o io.o scan.o \
minmax.o redistancing.o last_bit_float.o \
visc-aux.o simulation.o main.o \
velcontroller.o

ifeq ($(h5part),0)
NVCCFLAGS += -DNO_H5PART
endif

ifeq ($(h5),0)
NVCCFLAGS += -DNO_H5
endif

ifeq ($(datadump),0)
NVCCFLAGS += -D_NO_DUMPS_
endif
ifeq ($(datadump),1)
NVCCFLAGS += -D_SYNC_DUMPS_
endif

NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel)

# I have some device code in OBJS. I have to link it separatly
test:   $(OBJS) libcuda-dpd libcuda-rbc
	$(NVCC) $(NVCCFLAGS) $(OPT) -dlink -L../cuda-dpd/dpd -L../cuda-rbc/dpd $(OBJS) $(LIBS) -o gpuCode.o
	$(CXX)  $(LDFLAGS) gpuCode.o $(OBJS) $(LIBS) -o test

# a rule for the most of *.cu files
%.o:              %.cu;              $(NVCC) $(NVCCFLAGS) $(OPT)     $< -c -o $@

# some files must be compiled in diferent way (with `-dc' flag) See
# "Separate Compilation and Linking of CUDA C++ Device Code"
# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
contact.o:        contact.cu;        $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
containers.o:     containers.cu;     $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
dpd.o:            dpd.cu;            $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
fsi.o:            fsi.cu;            $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
last_bit_float.o: last_bit_float.cu; $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
visc-aux.o:       visc-aux.cu;       $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
wall.o:           wall.cu;           $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
include    Makefile.dep # dependencies

libcuda-dpd:
	make -C ../cuda-dpd/dpd libcuda-dpd.a \
	CXX="$(CXX)" NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" OPT="$(OPT)"

libcuda-rbc:
	make -C ../cuda-rbc libcuda-rbc.a \
	CXX="$(CXX)" NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" OPT="$(OPT)"

clean   : cleanall
cleanall:
	rm -f test *.o
	make -C ../cuda-rbc     clean
	make -C ../cuda-dpd     clean
	make -C ../cuda-dpd/dpd clean

.PHONY = clean cleanall libcuda-dpd libcuda-rbc
