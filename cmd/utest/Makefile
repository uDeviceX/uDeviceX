BIN = $(HOME)/bin
B = $(BIN)

P = util pack pack0

p = u.test

install:
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

.PHONY: install
