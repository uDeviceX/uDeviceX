BIN = $(HOME)/bin
MAN = $(HOME)/man

B = $(BIN)
M = $(MAN)

p = u
m = par

install:
	@echo install $p.$m
	@mkdir -p "$B"
	cp main   "$B/u.$m"
	cp main0  "$B/u.$m"0

doc: man
	@mkdir -p "$M/man1"
	cp u.$m.1 "$M/man1/u.$m.1"
man: u.$m.1

u.$m.1: main.adoc
	a2x --doctype manpage --format manpage --no-xmllint $<

.PHONY: install clean

clean:; rm -f u.$m.1

