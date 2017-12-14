BIN = $(HOME)/bin
B = $(BIN)

P = append awk awk0 bop bop0 circle util roi score score0

# prefix
p = u.punto

install:
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

.PHONY: install
