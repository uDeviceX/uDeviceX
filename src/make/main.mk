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
# DL: device link
# A: archive

PREFIX    = $(HOME)
BIN       = $(PREFIX)/bin
NVCC     ?= nvcc
ARCH     ?= -arch compute_35 -code sm_35
OPT	 ?= -O3 -g

CXXFLAGS  +=  -I$B -I$S
COMMON    +=  -std=c++11 $(OPT)

NCFLAGS    =           $(CXXFLAGS)
XCFLAGS    = $(COMMON) $(CXXFLAGS)
NVCCFLAGS += $(COMMON) -use_fast_math -restrict
LIBS      += -lcudart -lcurand -lnvToolsExt

LOG = @echo $< $@;
N  = $(LOG) $(NVCC)  $(ARCH) $(NVCCFLAGS)        --compiler-options '$(NCFLAGS)' $< -c -o $@
X  = $(LOG) $(NVCC)  -Wno-deprecated-gpu-targets --compiler-options '$(XCFLAGS)' $< -c -o $@
DL = $(LOG) $(NVCC) $(ARCH) -dlink `u.unmain $^` -o $@
L  = $(LOG) $(LINK) `u.main $O` $^ $(LIBS) -o $@
A  = $(LOG) ar r $@ `u.unmain $O` && ranlib $@

$B/udx: $B/libudx_cpu.a $B/libudx_cuda.a; $L
$B/gpu.o: $O; $(DL)

$B/libudx_cpu.a:  $O;       $A
$B/libudx_cuda.a: $B/gpu.o; $A

$O:  $B/.cookie
$B/.cookie:; $D; touch $@

clean:; -rm -f $B/udx $O $B/gpuCode.o $B/.cookie

install: $B/udx; cp $B/udx $(BIN)

test: install
	@echo log to atest.log
	@atest 2>&1 `find test -name main -type f`       | tee atest.log

.PHONY: clean test all D install
