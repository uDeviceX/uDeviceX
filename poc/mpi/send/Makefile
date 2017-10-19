MPICC = mpicc
MPIEXEC = mpiexec

main :  main.c; $(MPICC) $< -o $@
run: send
run:; $(MPIEXEC) -np 2 ./send

clean:; rm -f send
.PHONY: clean run
