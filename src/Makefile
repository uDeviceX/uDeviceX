include .cache.Makefile # local settings (see .cache.Makefile.* for examples)

NVCC     ?= nvcc
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35
OPT      ?= -O3 -use_fast_math -g
# for cuda-gdb: OPT=-O0 -g --device-debug

slevel   ?= -2

NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL) -std=c++11
LDFLAGS   += -L../cuda-dpd -L./bounce-back
NVCCFLAGS += -I../cuda-dpd -I./bounce-back -I.
LIBS      += -lcuda-dpd -lbounce-back -lcudart

OBJS = 	bund.o common.o dpd-forces.o glb.o io.o main.o minmax.o mesh.o tcells.o mrescue.o m.o

ifeq ($(h5),0)
NVCCFLAGS += -DNO_H5
endif

NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel)

# I have some device code in OBJS. I have to link it separatly
udx:   $(OBJS) libcuda-dpd libbounce-back
	$(NVCC) $(NVCCFLAGS) $(OPT) -dlink \
		-L../cuda-dpd -L./bounce-back $(OBJS) $(LIBS) -o gpuCode.o
	$(LINK)  $(LDFLAGS) gpuCode.o $(OBJS) $(LIBS) -o $@

# a rule for the most of *.cu files
%.o:              %.cu;              $(NVCC) $(NVCCFLAGS) $(OPT)     $< -c -o $@

# some files must be compiled in diferent way (with `-dc' flag) See
# "Separate Compilation and Linking of CUDA C++ Device Code"
# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
fsi.o:            fsi.cu;        $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
dpd-forces.o:     dpd-forces.cu; $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
bund.o:           bund.cu;       $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
minmax.o:         minmax.cu;     $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
mesh.o:           mesh.cu;       $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
tcells.o:         tcells.cu;     $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
mrescue.o:        mrescue.cu;    $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
glb.o:            glb.cu;        $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
include    Makefile.dep # dependencies

libcuda-dpd:; make -C ../cuda-dpd \
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" OPT="$(OPT)"

libbounce-back:; make -C ./bounce-back \
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" OPT="$(OPT)"

clean:  ; rm -f udx *.o
	make -C ../cuda-dpd clean
	make -C ./bounce-back clean

.PHONY = clean libcuda-dpd libbounce-back
