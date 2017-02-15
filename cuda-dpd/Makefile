libcuda-dpd.a: cuda-dpd.o cuda-dpd-bipartite.o cell-lists.o
	ar rcs $@ cuda-dpd.o cuda-dpd-bipartite.o cell-lists.o

# use `-dc' flag to do separate compilation
# Separate Compilation and Linking of CUDA C++ Device Code
# http://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code
cuda-dpd.o: cuda-dpd-floatized-1.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -o $@

cuda-dpd-bipartite.o: cuda-dpd-bipartite-floatized.cu cuda-dpd.h
	$(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -o $@

OPT_LOCAL=-O3 -g # I have to overwrite the optimization because this	\
                 # file fails to link with -O0 --device-debug

cell-lists.o: cell-lists.cu cell-lists.h
	$(NVCC) $(NVCCFLAGS) $(OPT_LOCAL) -c $< -o $@

../%.o:
	make -C ..
	LINK="$(LINK)" NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" OPT="$(OPT)"

clean:
	rm -f *.o *.a

subclean: clean

.PHONY = clean subclean celllists
