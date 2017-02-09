ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35

CXXFLAGS += -g -O2 -std=c++11

NVCCFLAGS += -arch=$(ARCH_VAL)  -code=$(CODE_VAL) -O4
NVCCFLAGS += --use_fast_math
NVCCFLAGS += -g

cell-lists.o: cell-lists.cu cell-lists.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cell-lists-faster.o: cell-lists-faster.cu cell-lists.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f test-cell-lists *.o

subclean: clean

.PHONY = clean subclean
