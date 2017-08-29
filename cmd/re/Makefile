BIN = $(HOME)/bin
B = $(BIN)

# prefix
p = u.re
P = push list clear peek peek0 peek1 conf

install:
	@echo install re tools to $B
	@mkdir -p $B
	for f in $P; do cp $p.$$f $B; done

.PHONY: install
