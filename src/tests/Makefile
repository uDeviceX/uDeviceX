include ../.cache.Makefile # local settings (see .cache.Makefile.* for examples)

NVCC     ?= nvcc
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35
OPT      ?= -O3 -use_fast_math -g
# for cuda-gdb: OPT=-O0 -g --device-debug

NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL) -std=c++11


bboxes: bboxes.cu
	$(NVCC) $(NVCCFLAGS) $(OPT) $< -o $@

# a rule for the most of *.cu files
%.o:              %.cu;              $(NVCC) $(NVCCFLAGS) $(OPT)     $< -c -o $@

clean:  ; rm -f bboxes *.o

.PHONY = clean
