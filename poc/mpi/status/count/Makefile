CC=mpicc
E=mpiexec
M=main
O=$M.o send.o recv.o

run: $M
run:; $E  -np 2 ./$M

$O: common.h
$M: $O

clean:; rm -f $M
.PHONY: clean run
