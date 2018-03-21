BIN = $(HOME)/bin
MAN = $(HOME)/man

B = $(BIN)
M = $(MAN)

p = u
m = var

install:
	@echo install $p.$m
	@mkdir -p "$B"
	cp util   "$B/$p.$m.util"
	cp main   "$B/$p.$m"

test:; atest test/*

.PHONY: install test
