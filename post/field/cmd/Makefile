BIN = $(HOME)/bin
B = $(BIN)

P = hdf2bov hdf2bov0

# prefix
p = u

install:
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

.PHONY: install
