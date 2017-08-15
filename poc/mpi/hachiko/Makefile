CC=mpicc.mpich
E=mpiexec.mpich
M=main

run: $M
run:; $E  -np 2 ./$M

clean:; rm -f $M
.PHONY: clean run
