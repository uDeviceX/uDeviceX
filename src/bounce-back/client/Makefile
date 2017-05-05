include ../../.cache.Makefile # local settings (see .cache.Makefile.* for examples)

NVCC     ?= nvcc
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35
OPT      ?= -O3 -use_fast_math -g -std=c++11
# for cuda-gdb: OPT=-O0 -g --device-debug


NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL)
LDFLAGS   += -L..
NVCCFLAGS += -I..
LIBS      += -lbounce-back -lcudart

all: referential

libbounce-back:; make -C .. \
	NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" OPT="$(OPT)"

referential: referential.o libbounce-back
	$(NVCC) $(NVCCFLAGS) $(OPT) -dlink \
		-L.. referential.o $(LIBS) -o gpuCode.o
	$(LINK) $(LDFLAGS) gpuCode.o referential.o $(LIBS) -o $@

%.o:              %.cu;              $(NVCC) $(NVCCFLAGS) $(OPT)     $< -c -o $@

clean:
	rm -rf *.o referential
