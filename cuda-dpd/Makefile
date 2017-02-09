cell-lists.o: cell-lists.cu cell-lists.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cell-lists-faster.o: cell-lists-faster.cu cell-lists.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f *.o

subclean: clean

.PHONY = clean subclean
