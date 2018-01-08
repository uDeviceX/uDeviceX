BIN = $(HOME)/bin
B = $(BIN)

P = ply2box ply2cm ply2pbc ply2rate ply2ext ply2ext0

# prefix
p = u

install:
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

.PHONY: install
