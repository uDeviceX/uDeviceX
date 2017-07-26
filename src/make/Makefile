all: $B/udx

NVCC     ?= nvcc
OPT      ?= -O3 -use_fast_math -g --compiler-options -Wall
# for cuda-gdb: OPT=-O0 -g --device-debug
ARCH     ?= -arch compute_35 -code sm_35

slevel   ?= -2

NVCCFLAGS += -I$B -I$S
NVCCFLAGS += -std=c++11
LIBS      += -lcudart

OBJ0 = $B/bund.o $B/mesh/collision.o $B/mesh/dist.o $B/mesh/bbox.o $B/diag.o $B/common.mpi.o \
	$B/dpd/local.o $B/field.o $B/glb.o $B/io.o  $B/m.o $B/main.o \
	$B/mbounce.o $B/mesh/props.o $B/mrescue.o $B/minmax.o \
	$B/solid.o $B/tcells.o $B/restart.o $B/dump.o $B/flu/int.o $B/flu/imp.o
OBJL = $B/clist/imp.o $B/l/linal.o $B/l/m.o $B/l/off.o $B/l/ply.o $B/scan/imp.o
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
C=$(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT)
D=mkdir `dirname $@` 2>/dev/null;     $C

# dev
$B/%.o:           $S/%.cu;            $C     $< -c -o $@
$B/l/%.o:         $S/l/%.cu;          $D     $< -c -o $@
$B/mesh/%.o:      $S/mesh/%.cu;       $D     $< -c -o $@
$B/dpd/local.o:   $S/dpd/local.cu;    $D     $< -c -o $@
$B/scan/imp.o:    $S/scan/imp.cu;     $D     $< -c -o $@
$B/clist/imp.o:   $S/clist/imp.cu;    $D     $< -c -o $@
$B/flu/%.o:       $S/flu/%.cu;        $D     $< -c -o $@

# dev and hst
$B/bund.o:  $S/bund.cu;      $C -dc $< -c -o $@
$B/glb.o:   $S/glb.cu;       $C -dc $< -c -o $@


clean:; -rm -f $B/udx $(OBJS) $B/gpuCode.o

.PHONY = clean all
