.PHONY: clean

clean:
	rm -f *.cudaobj
	rm -f *.o *.a

rbc-cuda.cu: rbc-cuda.h helper_math.h
libcuda-rbc.a: rbc-cuda.cudaobj
	ar rcs $@ $^

%.cudaobj:%.cu
	$(NVCC)  $(NVCCFLAGS) $(OPT) -c $< -o $@
