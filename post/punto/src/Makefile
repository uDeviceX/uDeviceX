BIN = $(HOME)/bin
B = $(BIN)

CFLAGS=-O3 -g -Wall -Wextra -std=c99 -pedantic
p = u
P = punto2dist
O = punto2dist.o

install: $P
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done
punto2dist: punto2dist.o

clean:; rm -f $P $O
.PHONY: clean install
