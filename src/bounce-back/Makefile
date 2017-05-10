libbounce-back.a: mbounce.o; ar rcs $@ $^
mbounce.o: mbounce.cu mbounce.h roots.h ../common.h ../.conf.h

# -dc flag: "Separate Compilation and Linking of CUDA C++ Device Code"
%.o:       %.cu;       $(NVCC) $(NVCCFLAGS) $(OPT) -dc $< -c -o $@

.PHONY: clean
clean:; rm -f *.o *.a
