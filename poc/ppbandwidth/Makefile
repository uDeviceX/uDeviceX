NVCC     = nvcc
ARCH_VAL = compute_35
CODE_VAL = sm_35
OPT      = -O3

NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL)
NVCCFLAGS += --compiler-options -Wall,-Wextra

LIBS      += -lcudart
PROGS      = pp

all: $(PROGS)

%: %.cu; $(NVCC) $(NVCCFLAGS) $(OPT) $< -o $@

pp: vanilla.h float4.h P4.h

.PHONY: all clean
clean:; rm -f $(PROGS)
