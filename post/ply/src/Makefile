BIN = $(HOME)/bin/
B = $(BIN)

CFLAGS=-O3 -Wall -Wextra -std=c99 -pedantic
ply2cm: ply2cm.c

clean:; rm ply2cm

install:
	mkdir -p $B
	cp ply2cm $B

.PHONY: clean install
