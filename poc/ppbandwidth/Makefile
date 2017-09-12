NVCC     = nvcc
ARCH_VAL = compute_35
CODE_VAL = sm_35
OPT      = -O0 --device-debug

NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL)
NVCCFLAGS += --compiler-options -Wall,-Wextra

LIBS      += -lcudart
PROGS      = pp

all: $(PROGS)

%: %.cu; $(NVCC) $(NVCCFLAGS) $(OPT) $< -o $@

.PHONY: all clean
clean:; rm -f $(PROGS)
