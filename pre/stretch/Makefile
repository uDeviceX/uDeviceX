P = main
BIN = $(HOME)/bin
B = $(BIN)

p = u
M = stretch

install:
	mkdir -p "$B"
	cp main  "$B/$p.$M"
	cp main0 "$B/$p.$M"0

test:; u.test test/*

.PHONY: install test
