all: $B/udx

NVCC     ?= nvcc
OPT      ?= -O3 -use_fast_math -g --compiler-options -Wall
# for cuda-gdb: OPT=-O0 -g --device-debug
ARCH     ?= -arch compute_35 -code sm_35

slevel   ?= -2

NVCCFLAGS += -I$B -I$S
NVCCFLAGS += -std=c++11
LIBS      += -lcudart

OBJ0 = $B/bund.o  $B/collision.o $B/diag.o $B/common.mpi.o $B/force.o $B/local.o \
	$B/field.o $B/glb.o $B/io.o  $B/m.o $B/main.o \
	$B/mbounce.o $B/mesh.o $B/mrescue.o $B/minmax.o \
	$B/solid.o $B/tcells.o $B/restart.o $B/dump.o
OBJL = $B/l/clist.o $B/l/linal.o $B/l/m.o $B/l/off.o $B/l/ply.o $B/l/scan.o $B/l/rnd.o
OBJS = $(OBJ0) $(OBJL)

ifeq ($(h5),0)
NVCCFLAGS += -DNO_H5
endif

NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel)

# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
$B/udx: $(OBJS)
	$(NVCC)  $(ARCH) -dlink $(OBJS) $(NVCCLIBS) -o $B/gpuCode.o
	$(LINK)  $B/gpuCode.o $(OBJS) $(LIBS) -o $@

# compile commands
0=$(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT)
1=mkdir $B/l   2>/dev/null; $0

# dev
$B/%.o:     $S/%.cu;         $0     $< -c -o $@
$B/l/%.o:   $S/l/%.cu;       $1     $< -c -o $@

# dev and hst
$B/force.o: $S/force.cu;     $0 -dc $< -c -o $@
$B/bund.o:  $S/bund.cu;      $0 -dc $< -c -o $@
$B/glb.o:   $S/glb.cu;       $0 -dc $< -c -o $@
$B/local.o: $S/dpd/local.cu; $0 -dc $< -c -o $@

# dev and hst
$B/l/rnd.o: $S/l/rnd.cu;     $1 -dc $< -c -o $@

clean:; -rm -f $B/udx $(OBJS) $B/gpuCode.o

.PHONY = clean all
