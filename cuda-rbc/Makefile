.PHONY: clean

clean:
	rm -f *.cudaobj
	rm -f *.o *.a

libcuda-rbc.a: rbc-cuda.cudaobj
	ar rcs $@ $^

%.cudaobj:%.cu
	$(NVCC)  $(NVCCFLAGS)  $(INCLUDES) -c $< -o $@
