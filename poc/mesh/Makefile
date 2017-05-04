include ../../src/.cache.Makefile

NVCC     ?= nvcc
ARCH_VAL ?= compute_35
CODE_VAL ?= sm_35
OPT      ?= -O3 -use_fast_math -g

NVCCFLAGS += -arch $(ARCH_VAL) -code $(CODE_VAL) -std=c++11

default: ply_io inmesh

ply_io: ply_io.cu mesh.h
	$(NVCC) $(NVCCFLAGS) $(OPT) $< -o $@

inmesh: inmesh.cu mesh.h collision.h
	$(NVCC) $(NVCCFLAGS) $(OPT) $< -o $@

clean:
	rm -rf ply_io inmesh
