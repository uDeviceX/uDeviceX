BIN = $(HOME)/bin/
B = $(BIN)

CFLAGS=-O3 -Wall -Wextra -std=c99 -pedantic
p = u
P = ply2cm0   ply2box0
O = ply2cm0.o ply2box0.o

install: $P
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done
ply2box0: ply2box0.o
ply2cm0 : ply2cm0.o

clean:; rm -f $P $O
.PHONY: clean install
