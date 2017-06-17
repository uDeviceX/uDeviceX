all: udx

NVCC     ?= nvcc
OPT      ?= -O3 -use_fast_math -g --compiler-options -Wall
# for cuda-gdb: OPT=-O0 -g --device-debug
ARCH     ?= -arch compute_35 -code sm_35

slevel   ?= -2

NVCCFLAGS += -std=c++11 -I. -I$S
LIBS      += -lcudart

OBJ0 = bund.o  collision.o common.o force.o dpd.o field.o glb.o io.o  m.o main.o \
	mbounce.o mesh.o mrescue.o ply.o minmax.o solid.o tcells.o restart.o
OBJL = l.clist.o l.m.o l.off.o l.scan.o
OBJS = $(OBJ0) $(OBJL)

ifeq ($(h5),0)
NVCCFLAGS += -DNO_H5
endif

NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel)

# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
udx: $(OBJS)
	$(NVCC)  $(ARCH) -dlink $(OBJS) $(NVCCLIBS) -o gpuCode.o
	$(LINK)  gpuCode.o $(OBJS) $(LIBS) -o $@

# dev
%.o:     $S/%.cu;         $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT)     $< -c -o $@
l.%.o:   $S/l/%.cu;       $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT)     $< -c -o $@

# dev and hst
force.o: $S/force.cu;     $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
bund.o:  $S/bund.cu;      $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
glb.o:   $S/glb.cu;       $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
dpd.o:   $S/dpd/dpd.cu;   $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@

clean:; -rm -f udx $(OBJS) gpuCode.o

.PHONY = clean all
