NVCC     = nvcc
ARCH_VAL = compute_35
CODE_VAL = sm_35
OPT      = -O2

NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL)
NVCCFLAGS += --compiler-options -Wall,-Wextra

LIBS   += -lcudart
M      = main

all: $M

%: %.cu; $(NVCC) $(NVCCFLAGS) $(OPT) $< -o $@

.PHONY: all clean
clean:; rm -f $M
