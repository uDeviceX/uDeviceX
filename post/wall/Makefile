BIN = $(HOME)/bin
B = $(BIN)

P = dump dump0

# prefix
p = u.wall

install:
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

.PHONY: install
