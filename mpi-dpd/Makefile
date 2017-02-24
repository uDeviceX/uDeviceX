include .cache.Makefile # local settings (see .cache.Makefile.* for examples)

NVCC     ?= nvcc
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35
OPT      ?= -O3 -use_fast_math -g
# for cuda-gdb: OPT=-O0 -g --device-debug

slevel   ?= -2

NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL)
LDFLAGS   += -L../cuda-dpd -L../cuda-rbc -L../last-bit
NVCCFLAGS += -I../cuda-dpd -I../cuda-rbc -I../last-bit -I../argument-parser
LIBS      += -lcuda-dpd -lcuda-rbc -llast-bit -lcudart

OBJS = 	common.o io.o minmax.o dpd-forces.o bund.o main.o

ifeq ($(h5part),0)
NVCCFLAGS += -DNO_H5PART
endif

ifeq ($(h5),0)
NVCCFLAGS += -DNO_H5
endif

NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel)

# I have some device code in OBJS. I have to link it separatly
test:   $(OBJS) libcuda-dpd libcuda-rbc liblast-bit
	$(NVCC) $(NVCCFLAGS) $(OPT) -dlink \
		-L../cuda-dpd -L../cuda-rbc -L../last-bit $(OBJS) $(LIBS) -o gpuCode.o
	$(LINK)  $(LDFLAGS) gpuCode.o $(OBJS) $(LIBS) -o test

# a rule for the most of *.cu files
%.o:              %.cu;              $(NVCC) $(NVCCFLAGS) $(OPT)     $< -c -o $@

# some files must be compiled in diferent way (with `-dc' flag) See
# "Separate Compilation and Linking of CUDA C++ Device Code"
# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
fsi.o:            fsi.cu;            $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
dpd-forces.o:     dpd-forces.cu;     $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
bund.o:           bund.cu;     $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
include    Makefile.dep # dependencies

libcuda-dpd:; make -C ../cuda-dpd \
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" OPT="$(OPT)"

libcuda-rbc:; make -C ../cuda-rbc \
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" OPT="$(OPT)"

liblast-bit:; make -C ../last-bit \
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" OPT="$(OPT)"

clean   : cleanall
cleanall:; rm -f test *.o
	make -C ../cuda-rbc clean
	make -C ../cuda-dpd clean
	make -C ../last-bit clean

.PHONY = clean cleanall libcuda-dpd libcuda-rbc liblast-bit
