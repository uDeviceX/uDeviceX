ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35

CXXFLAGS += -g -O2 -std=c++11

NVCCFLAGS += -arch=$(ARCH_VAL)  -code=$(CODE_VAL) -O4
NVCCFLAGS += --use_fast_math

ifeq "$(profile)" "1"
NVCCFLAGS += -lineinfo -D_PROFILE_ 
CXXFLAGS += -D_PROFILE_ -g
endif

ifneq "$(debug)" "1"
NVCCFLAGS += -DNDEBUG
else
NVCCFLAGS += -g
endif

CELLING=cell-lists-faster.cu
ifeq "$(legacy)" "1"
CELLING=cell-lists.cu
endif

profiler-dpd.o:	profiler-dpd.cpp profiler-dpd.h
	$(NVCC) $(CXXFLAGS) -c profiler-dpd.cpp

cell-lists.o: cell-lists.cu cell-lists.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cell-lists-faster.o: cell-lists-faster.cu cell-lists.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f test-cell-lists *.o

subclean: clean

.PHONY = clean subclean
