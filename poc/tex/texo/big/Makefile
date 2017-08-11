NVCC     = nvcc
ARCH_VAL = compute_35
CODE_VAL = sm_35
OPT      = -O0 -g --device-debug

NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL)
NVCCFLAGS += --compiler-options -Wall,-Wextra

LIBS   += -lcudart
M      = main

all: $M
$M.o: u.h texo.h

%: %.cu; $(NVCC) $(NVCCFLAGS) $(OPT) $< -o $@

.PHONY: all clean
clean:; rm -f $M
