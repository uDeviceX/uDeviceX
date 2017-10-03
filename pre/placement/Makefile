 # where to install
BIN = $(HOME)/bin
B = $(BIN)

p = plcmt
P = p2mat hex select ro

install:;
	mkdir -p $(BIN)
	for f in $P; do \
	    echo cp $$f $B/$p.$$f; \
        done

.PHONY: install
