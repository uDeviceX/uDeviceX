BIN = $(HOME)/bin
B = $(BIN)

P = append util

# prefix
p = u.punto

install:
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

test:; atest test/*

.PHONY: install test
