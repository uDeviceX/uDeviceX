MPICC = mpicc
MPIEXEC = mpiexec

send :  send.c; $(MPICC) $< -o $@
run: send
run:; $(MPIEXEC) -np 2 ./send

clean:; rm -f send
.PHONY: clean run
