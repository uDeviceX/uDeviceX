.PHONY: clean

clean:
	rm -f *.o *.a

rbc-cuda.cu: rbc-cuda.h helper_math.h
libcuda-rbc.a: rbc-cuda.o
	ar rcs $@ $^

%.o :%.cu; $(NVCC)  $(NVCCFLAGS) $(OPT) -c $< -o $@
