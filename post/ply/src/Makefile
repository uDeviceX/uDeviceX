BIN = $(HOME)/bin/
B = $(BIN)

CFLAGS=-O3 -Wall -Wextra -std=c99 -pedantic
p = u
P = ply2cm0
install: $P
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

clean:; rm ply2cm
.PHONY: clean install
