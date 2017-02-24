liblast-bit.a: last-bit.o; ar rcs $@ $^
last-bit.o: last-bit.h

# -dc flag: "Separate Compilation and Linking of CUDA C++ Device Code"
%.o:       %.cu;       $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@

.PHONY: clean
clean:; rm -f *.o *.a
