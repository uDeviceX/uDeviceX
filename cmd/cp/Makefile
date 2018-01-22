BIN = $(HOME)/bin
B = $(BIN)

P = rbc rbc0 util

p = u.cp

install:
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

.PHONY: install
