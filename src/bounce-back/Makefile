libbounce-back.a: sbounce.o; ar rcs $@ $^
sbounce.o: sbounce.cu sbounce.h bbshapes.impl.h ../common.h ../.conf.h

# -dc flag: "Separate Compilation and Linking of CUDA C++ Device Code"
%.o:       %.cu;       $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@

.PHONY: clean
clean:; rm -f *.o *.a
