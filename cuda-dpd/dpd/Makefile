include ../Makefile

CUDADPD = cuda-dpd-floatized-1.cu
CUDADPDBIP = cuda-dpd-bipartite-floatized.cu
NVCCFLAGS += -DVISCOSITY_S_LEVEL=$(slevel) -lineinfo -Xptxas -v

.DEFAULT_GOAL = test-dpd

libcuda-dpd.a: cuda-dpd.o cuda-dpd-bipartite.o celllists
	ar rcs $@ cuda-dpd.o cuda-dpd-bipartite.o ../cell-lists.o ../cell-lists-faster.o

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
	rm -f test-dpd *.o *.a
	make -C ../ clean

subclean: clean

.PHONY = clean subclean celllists
