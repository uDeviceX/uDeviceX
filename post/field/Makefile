P = avg avg0
BIN = $(HOME)/bin
B = $(BIN)

install:
	mkdir -p "$B"
	for f in $P; do cp $$f "$B"/u.$$f; done

test:; u.test test/*

.PHONY: install test
