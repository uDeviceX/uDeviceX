 # where to install
BIN = $(HOME)/bin
B = $(BIN)

p = plcmt
P = p2mat hex select ro unmat util

install:;
	mkdir -p $(BIN)
	for f in $P; do \
	    cp $$f $B/$p.$$f; \
	done

test:; u.test $P

.PHONY: install test
