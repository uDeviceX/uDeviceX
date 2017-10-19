MPICC = mpicc
MPIEXEC = mpiexec
M = main

$M :  $M.c; $(MPICC) $< -o $@

run: $M
run:; $(MPIEXEC) -np 2 ./$M

clean:; rm -f $M
.PHONY: clean run
