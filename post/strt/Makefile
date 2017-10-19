BIN = $(HOME)/bin
B = $(BIN)

P = cp

# prefix
p = u.strt

install:
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

test:; u.test test/*

.PHONY: install test
