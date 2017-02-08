-include .cache.Makefile

include ../Makefile

CUDADPD = cuda-dpd.cu
CUDADPDBIP = cuda-dpd-bipartite.cu

ifeq "$(floatized)" "1"
CUDADPD = cuda-dpd-floatized-1.cu
CUDADPDBIP = cuda-dpd-bipartite-floatized.cu
else
ifeq "$(floatized)" "2"
CUDADPD = cuda-dpd-floatized-2.cu
CUDADPDBIP = cuda-dpd-bipartite-floatized.cu
endif
endif

ifeq "$(flops)" "1"
        NVCCFLAGS += -D_COUNT_FLOPS
endif

ifeq "$(nonperiodic)" "1"
    NVCCFLAGS += -D_NONPERIODIC_KERNEL_
endif

ifeq "$(MAKECMDGOALS)" "libcuda-dpd.so"
CXXFLAGS += -fPIC
NVCCFLAGS += -Xcompiler -fpic
endif

NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel) -lineinfo -Xptxas -v

.DEFAULT_GOAL = test-dpd

libcuda-dpd.a: cuda-dpd.o cuda-dpd-bipartite.o ../profiler-dpd.o celllists
	ar rcs $@ cuda-dpd.o cuda-dpd-bipartite.o ../profiler-dpd.o ../cell-lists.o ../cell-lists-faster.o

libcuda-dpd.so: cuda-dpd.o cuda-dpd-bipartite.o ../profiler-dpd.o celllists
	$(CXX) $(CXXFLAGS) -shared cuda-dpd.o cuda-dpd-bipartite.o ../profiler-dpd.o ../cell-lists.o ../cell-lists-faster.o -o libcuda-dpd.so  -lcudart -lcurand

# use `-dc' flag to do separate compilation
# Separate Compilation and Linking of CUDA C++ Device Code
# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
cuda-dpd.o: $(CUDADPD) cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -dc -cubin $(CUDADPD)
	$(NVCC) $(NVCCFLAGS) -dc $(CUDADPD) -o $@

cuda-dpd-bipartite.o: $(CUDADPDBIP) cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -dc $(CUDADPDBIP) -o $@

../%.o:
	make -C ../ $(@:../%=%) CXX="$(CXX)" NVCC="$(NVCC)"

celllists:
	make -C ../ cell-lists-faster.o NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)"
	make -C ../ cell-lists.o NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)"

clean:
	rm -f test-dpd libcuda-dpd.so *.o *.a
	make -C ../ clean

subclean: clean

.PHONY = clean subclean celllists
