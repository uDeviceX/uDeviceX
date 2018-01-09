BIN = $(HOME)/bin
B = $(BIN)

P = append bop bop0 circle disp disp0 util roi roi0 score score0

# prefix
p = u.punto

install:
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

.PHONY: install
