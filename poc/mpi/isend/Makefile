cc=mpicc

isend :  isend.c; $(cc) $< -o $@
run: isend
run:; mpiexec -np 2 ./isend

clean:; rm -f isend
.PHONY: clean run
