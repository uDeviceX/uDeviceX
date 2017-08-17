# a fragment of makefile
# see src/make/ and tools/udeps
# B: binary
# S: source
# O: objects
# target D: directories

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

$B/udx: $O
	$(NVCC)  $(ARCH) -dlink $O $(NVCCLIBS) -o $B/gpuCode.o
	$(LINK)  $B/gpuCode.o $O $(LIBS) -o $@

N  = $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT)     $< -c -o $@
NN = $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@

# special rule
$B/glb.o: $S/glb.cu; $(NN)
clean:; -rm -f $B/udx $O $B/gpuCode.o

.PHONY = clean all D
