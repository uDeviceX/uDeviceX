all: $B/udx

NVCC     ?= nvcc
OPT      ?= -O3 -use_fast_math -g --compiler-options -Wall
# for cuda-gdb: OPT=-O0 -g --device-debug
ARCH     ?= -arch compute_35 -code sm_35

slevel   ?= -2

NVCCFLAGS += -I$B -I$S
NVCCFLAGS += -std=c++11
LIBS      += -lcudart

OBJ0 = $B/bund.o  $B/collision.o $B/common.o $B/force.o $B/local.o \
	$B/field.o $B/glb.o $B/io.o  $B/m.o $B/main.o \
	$B/mbounce.o $B/mesh.o $B/mrescue.o $B/minmax.o \
	$B/solid.o $B/tcells.o $B/restart.o
OBJL = $B/l.clist.o $B/l.gsl.o $B/l.m.o $B/l.off.o $B/l.ply.o $B/l.scan.o $B/l.rnd.o
OBJS = $(OBJ0) $(OBJL)

ifeq ($(h5),0)
NVCCFLAGS += -DNO_H5
endif

NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel)

# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
$B/udx: $(OBJS)
	$(NVCC)  $(ARCH) -dlink $(OBJS) $(NVCCLIBS) -o $B/gpuCode.o
	$(LINK)  $B/gpuCode.o $(OBJS) $(LIBS) -o $@

# dev
$B/%.o:     $S/%.cu;         $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT)     $< -c -o $@
$B/l.%.o:   $S/l/%.cu;       $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT)     $< -c -o $@

# dev and hst
$B/force.o: $S/force.cu;     $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
$B/bund.o:  $S/bund.cu;      $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
$B/glb.o:   $S/glb.cu;       $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
$B/local.o: $S/dpd/local.cu; $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@

# dev and hst
$B/l.rnd.o: $S/l/rnd.cu;   $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@

clean:; -rm -f $B/udx $(OBJS) $B/gpuCode.o

.PHONY = clean all
