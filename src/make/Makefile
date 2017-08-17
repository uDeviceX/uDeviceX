# a fragment of makefile
# see src/make/ and tools/udeps
# B: binary
# S: source
# O: objects
# D: directories

all: $B/udx
$O: D

NVCC     ?= nvcc
OPT      ?= -O3 -use_fast_math -g --compiler-options -Wall,-Wextra
ARCH     ?= -arch compute_35 -code sm_35

slevel   ?= -2

NVCCFLAGS += -I$B -I$S
NVCCFLAGS += -std=c++11
LIBS      += -lcudart

ifeq ($(h5),0)
NVCCFLAGS += -DNO_H5
endif

NVCCFLAGS += -DS_LEVEL=$(slevel)

# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
$B/udx: $O
	$(NVCC)  $(ARCH) -dlink $O $(NVCCLIBS) -o $B/gpuCode.o
	$(LINK)  $B/gpuCode.o $O $(LIBS) -o $@

# compile commands
C=$(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT)

# dev
$B/%.o:           $S/%.cu;            $C     $< -c -o $@
$B/l/%.o:         $S/l/%.cu;          $C     $< -c -o $@
$B/mesh/%.o:      $S/mesh/%.cu;       $C     $< -c -o $@
$B/io/%.o:        $S/io/%.cu;         $C     $< -c -o $@
$B/dpd/local.o:   $S/dpd/local.cu;    $C     $< -c -o $@
$B/wall/exch.o:   $S/wall/exch.cu;    $C     $< -c -o $@
$B/d/api.o:       $S/d/api.cu;        $C     $< -c -o $@

# generic interface and implementation rules
$B/%/int.o:       $S/%/int.cu;        $C     $< -c -o $@
$B/%/imp.o:       $S/%/imp.cu;        $C     $< -c -o $@

# dev and hst
$B/bund.o:  $S/bund.cu;      $C -dc $< -c -o $@
$B/glb.o:   $S/glb.cu;       $C -dc $< -c -o $@

clean:; -rm -f $B/udx $O $B/gpuCode.o

.PHONY = clean all D
