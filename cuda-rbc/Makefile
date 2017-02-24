libcuda-rbc.a: rbc-cuda.o
	ar rcs $@ $^

rbc-cuda.o: rbc-cuda.h helper_math.h
%.o :%.cu; $(NVCC)  $(NVCCFLAGS) $(OPT) -c $< -o $@

.PHONY: clean
clean:; rm -f *.o *.a
