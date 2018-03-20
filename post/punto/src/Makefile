CC=gcc
BIN = $(HOME)/bin
B = $(BIN)

CFLAGS=-O0 -g -Wall -Wextra -std=c99 -pedantic
LIB = -lm
p = u
P = punto2dist
O = punto2dist.o

install: $P
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

punto2dist: punto2dist.o; $(CC) -o $@ $< $(LIB)

clean:; rm -f $P $O
.PHONY: clean install
