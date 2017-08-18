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
ARCH     ?= -arch compute_35 -code sm_35

CXXFLAGS += -O3 -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -I$B -I$S
CXXFLAGS += -DS_LEVEL=-2

NVCCFLAGS += -O3 -use_fast_math -g
NVCCFLAGS += -std=c++11
LIBS      += -lcudart

N  = $(NVCC)  $(ARCH) $(NVCCFLAGS) --compiler-options '$(CXXFLAGS)'     $< -c -o $@
X  = $(NVCC)                       --compiler-options '$(CXXFLAGS)'     $< -c -o $@
NN = $(NVCC)  $(ARCH) $(NVCCFLAGS) --compiler-options '$(CXXFLAGS)' -dc $< -c -o $@
L  = $(NVCC)  $(ARCH) -dlink $O $(NVCCLIBS) -o $B/gpuCode.o && \
	$(LINK)  $B/gpuCode.o $O $(LIBS) -o $@

all: $B/udx
$O:  $B/.cookie
$B/.cookie:;       $D ; touch $@
$B/udx: $O; $L

# special rules
$B/glb.o: $S/glb.cu; $(NN)
$B/bund.o: $S/bund.cu; $(NN)
clean:; -rm -f $B/udx $O $B/gpuCode.o $B/.cookie

.PHONY = clean all D
