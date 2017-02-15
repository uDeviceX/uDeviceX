OPT_LOCAL=-O3 -g # I have to overwrite the optimization because this	\
                 # file fails to link with -O0 --device-debug
cell-lists.o: cell-lists.cu cell-lists.h
	$(NVCC) $(NVCCFLAGS) $(OPT_LOCAL) -c $< -o $@

clean: ; rm -f *.o
subclean: clean
.PHONY = clean subclean
