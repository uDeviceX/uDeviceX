# a fragment of makefile
# see src/make/ and tools/udeps
# B: binary
# S: source
# O: objects

# commands
# D: create directories
# N: nvcc compile
# NN: special nvcc compile command
# X: c++ compile
# L: link

NVCC     ?= nvcc
OPT      ?= -O3 -use_fast_math -g --compiler-options -Wall,-Wextra
ARCH     ?= -arch compute_35 -code sm_35
slevel   ?= -2

NVCCFLAGS += -I$B -I$S
NVCCFLAGS += -std=c++11
NVCCFLAGS += -DS_LEVEL=$(slevel)
LIBS      += -lcudart

N  = $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT)     $< -c -o $@
X  = $N
NN = $(NVCC) $(ARCH) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@
L  = $(NVCC)  $(ARCH) -dlink $O $(NVCCLIBS) -o $B/gpuCode.o && \
	$(LINK)  $B/gpuCode.o $O $(LIBS) -o $@

all: $B/udx
$O:  $B/.cookie
$B/.cookie: $D; touch $@
$B/udx: $O; $L

# special rules
$B/glb.o: $S/glb.cu; $(NN)
$B/bund.o: $S/bund.cu; $(NN)
clean:; -rm -f $B/udx $O $B/gpuCode.o $B/.cookie

.PHONY = clean all D
