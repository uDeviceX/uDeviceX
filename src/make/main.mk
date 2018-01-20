# a fragment of makefile
# see src/make/ and tools/udeps
# B: binary dir
# S: source dir
#
# O: list of objects

# commands
# D: create directories
# N: nvcc compile
# X: c++ compile
# L: link

NVCC     ?= nvcc
ARCH     ?= -arch compute_35 -code sm_35
OPT	 ?= -O3 -g

CXXFLAGS  +=  -I$B -I$S
COMMON    +=  -std=c++11 $(OPT)

NCFLAGS    =           $(CXXFLAGS)
XCFLAGS    = $(COMMON) $(CXXFLAGS)
NVCCFLAGS += $(COMMON) -use_fast_math -restrict
LIBS      += -lcudart -lcurand

LOG = @echo $< $@;
N  = $(LOG) $(NVCC)  $(ARCH) $(NVCCFLAGS) --compiler-options '$(NCFLAGS)'     $< -c -o $@
X  = $(LOG) $(NVCC)  $(ARCH)              --compiler-options '$(XCFLAGS)'     $< -c -o $@
L  = $(LOG) $(NVCC)  $(ARCH) -dlink $O $(NVCCLIBS) -o $B/gpuCode.o && \
	$(LINK)  $B/gpuCode.o $O $(LIBS) -o $@

$B/udx: $O; $L
$O:  $B/.cookie
$B/.cookie:; $D; touch $@

clean:; -rm -f $B/udx $O $B/gpuCode.o $B/.cookie

test:
	@echo log to atest.log
	@atest 2>&1 `find test -type f`       | tee atest.log


.PHONY: clean test all D
