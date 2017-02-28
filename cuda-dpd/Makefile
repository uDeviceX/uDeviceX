libcuda-dpd.a: cuda-dpd.o cuda-dpd-bipartite.o cell-lists.o
	ar rcs $@ cuda-dpd.o cuda-dpd-bipartite.o cell-lists.o

# use `-dc' flag to do separate compilation
# Separate Compilation and Linking of CUDA C++ Device Code
# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
cuda-dpd.o: cuda-dpd-floatized-1.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -o $@

cuda-dpd-bipartite.o: cuda-dpd-bipartite-floatized.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -o $@

OPT_LOCAL=-O3 -g # I have to overwrite the flags because this	\
                 # file fails to link with -O0 --device-debug
                 # error message:
                 # ptxas fatal   : Unresolved extern function 'cudaDeviceSynchronize'
cell-lists.o: cell-lists.h
cell-lists.o: cell-lists.cu
	$(NVCC) $(NVCCFLAGS) $(OPT_LOCAL) -c $< -o $@

.PHONY = clean
clean:; rm -f *.o *.a

