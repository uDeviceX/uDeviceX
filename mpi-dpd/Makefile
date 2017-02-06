-include .cache.Makefile

NVCC ?= nvcc -ccbin $(CXX)
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35

NVCCFLAGS += -I$(HDF5_DIR)/include -Xcudafe "--diag_suppress=unrecognized_gcc_pragma"
NVCCFLAGS +=  -arch $(ARCH_VAL) -code $(CODE_VAL) -O3 -use_fast_math -g
#-DNDEBUG
CXXFLAGS += -L../cuda-dpd/dpd -L../cuda-rbc/ -L../cuda-ctc/ -O3 -g  -std=c++11 -DNDEBUG
NVCCFLAGS += -I../cuda-dpd/dpd -I../cuda-rbc/ -I../cuda-ctc
#NVCCFLAGS +=  -DREPORT_TOPOLOGY $(CRAY_PMI_INCLUDE_OPTS)
#NVCCFLAGS += -DCUSTOM_REORDERING=1

OBJS = 	containers.o wall.o dpd.o fsi.o contact.o \
	redistribute-particles.o redistribute-rbcs.o \
	solvent-exchange.o solute-exchange.o \
	common.o io.o \
	scan.o minmax.o redistancing.o \
	last_bit_float.o \
	visc-aux.o \
	simulation.o main.o velcontroller.o

LIBS += -lcuda-dpd -lcuda-rbc -lcuda-ctc -lcudart -ldl -lz -lpthread

nvtx ?= 0
ifneq "$(nvtx)" "0"
LIBS += -lnvToolsExt
NVCCFLAGS += -D_USE_NVTX_ -lineinfo
endif

ifneq "$(h5part)" "0"
LIBS += -L${H5PART_DIR}/lib -lH5Part
else
NVCCFLAGS += -DNO_H5PART
endif

ifneq "$(h5)" "0"
LIBS += -L$(HDF5_DIR)/lib -lhdf5
else
NVCCFLAGS += -DNO_H5
endif

vtk ?= 0
ifneq "$(vtk)" "0"
LIBS += -lvtkIOImage-6.2 -lvtkCommonDataModel-6.2 -lvtkpng-6.2  -lvtktiff-6.2 -lvtkmetaio-6.2 -lvtkDICOMParser-6.2 -lvtkzlib-6.2 -lvtksys-6.2 -lvtkIOXMLParser-6.2 -lvtkCommonExecutionModel-6.2 -lvtkCommonTransforms-6.2  -lvtkCommonCore-6.2 -lvtkIOXML-6.2 -lvtkexpat-6.2  -lvtkjpeg-6.2 -lvtkIOCore-6.2  -lvtkCommonSystem-6.2 -lvtkCommonTransforms-6.2 -lvtkCommonMath-6.2 -lvtkCommonMisc-6.2
else
NVCCFLAGS += -DNO_VTK
endif

floatized ?= 1
slevel ?= 0
flops ?= 0
datadump ?= 1

ifeq "$(datadump)" "0"
NVCCFLAGS += -D_NO_DUMPS_
endif
ifeq "$(datadump)" "1"
NVCCFLAGS += -D_SYNC_DUMPS_
endif
ifeq "$(datadump)" "2"
NVCCFLAGS += -D_ASYNC_DUMPS_
endif

NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel)

inquire: $(bash [ `cat slevel.txt` == "$(slevel)" ] || { echo "cleanall" ; } )
	$(bash echo "$(slevel)" > slevel.txt )
	$(MAKE) $(MFLAGS) test

test:  $(OBJS) libcuda-dpd libcuda-rbc libcuda-ctc
	echo h5part is "$(h5part)"
	$(NVCC) $(NVCCFLAGS) -dlink $(OBJS) $(LIBS) -o gpuCode.o # I have some device code in OBJS. I have to link it separatly
	$(CXX)  $(CXXFLAGS) gpuCode.o $(OBJS) $(LIBS) -o test


-include $(OBJS:.o=.d)

# use `-dc' flag to do separate compilation of `last_bit_float'
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
#	$(NVCC) $(NVCCFLAGS) -c $*.cu --cubin
#	$(NVCC) $(NVCCFLAGS) -c $*.cu --ptx

libcuda-dpd:
	$(MAKE) -C ../cuda-dpd/dpd libcuda-dpd.a nonperiodic=1 floatized="$(floatized)" flops="$(flops)" CXX="$(CXX)" NVCC="$(NVCC)" ARCH_VAL="$(ARCH_VAL)" CODE_VAL="$(CODE_VAL)" slevel="$(slevel)"

libcuda-rbc:
	$(MAKE) -C ../cuda-rbc/ libcuda-rbc.a CXX="$(CXX)" NVCC="$(NVCC)" ARCH_VAL="$(ARCH_VAL)" CODE_VAL="$(CODE_VAL)"

libcuda-ctc:
	$(MAKE) -C ../cuda-ctc/ libcuda-ctc.a CXX="$(CXX)" NVCC="$(NVCC)" ARCH_VAL="$(ARCH_VAL)" CODE_VAL="$(CODE_VAL)"

clean:
	rm -f test *.o *.d slevel.txt

cleanall: clean
	rm -f slevel.txt
	make -C ../cuda-dpd/dpd clean
	make -C ../cuda-rbc clean
	make -C ../cuda-ctc clean

#we want to give always a try to update the libs - too complicated to track dependecies there
.PHONY = clean libcuda-dpd libcuda-rbc libcuda-ctc inquire
