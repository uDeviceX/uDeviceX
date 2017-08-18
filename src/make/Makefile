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

CXXFLAGS +=  -I$B -I$S
COMMON     =  -std=c++11 -O2 -g
NCFLAGS    =           $(CXXFLAGS)
XCFLAGS   += $(COMMON) $(CXXFLAGS)
NVCCFLAGS += $(COMMON) -use_fast_math 
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
