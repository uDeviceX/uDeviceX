include ../Makefile

libcuda-dpd.a: cuda-dpd.o cuda-dpd-bipartite.o celllists
	ar rcs $@ cuda-dpd.o cuda-dpd-bipartite.o ../cell-lists.o ../cell-lists-faster.o

# use `-dc' flag to do separate compilation
# Separate Compilation and Linking of CUDA C++ Device Code
# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
cuda-dpd.o: cuda-dpd-floatized-1.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

cuda-dpd-bipartite.o: cuda-dpd-bipartite-floatized.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

../%.o:
	make -C ../ $(@:../%=%) \
	CXX="$(CXX)" NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)"

celllists:
	make -C ../ cell-lists-faster.o \
	CXX="$(CXX)" NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)"

	make -C ../ cell-lists.o \
	CXX="$(CXX)" NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)"

clean:
	rm -f *.o *.a

subclean: clean

.PHONY = clean subclean celllists
