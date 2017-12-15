BIN = $(HOME)/bin
B = $(BIN)

P = off off0 ply ply0

# prefix
p = u.gnuplot

install:
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

.PHONY: install
