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

OPT      = -O3  -std=c++98
COMMON   = $(CXXFLAGS) -I$B -I$S -DS_LEVEL=-2
NCFLAGS  = $(COMMON)
XCFLAGS += $(COMMON) $(OPT) -Wall -Wextra
NVCCFLAGS += $(OPT) -use_fast_math -g
LIBS      += -lcudart

N  = $(NVCC)  $(ARCH) $(NVCCFLAGS) --compiler-options '$(NCFLAGS)'     $< -c -o $@
X  = $(NVCC)  $(ARCH)              --compiler-options '$(XCFLAGS)'     $< -c -o $@
NN = $(NVCC)  $(ARCH) $(NVCCFLAGS) --compiler-options '$(NCFLAGS)' -dc $< -c -o $@
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
