P = dawk dawk0
BIN = $(HOME)/bin
B = $(BIN)

p = u

install:
	mkdir -p "$B"
	for f in $P; do cp $$f "$B"/$p.$$f; done

test:; u.test test/*

.PHONY: install test
