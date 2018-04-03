BIN = $(HOME)/bin
B = $(BIN)

P = 2vtk 2vtk0 append bop bop0 cat circle disp disp0 util ro roi roi0 score score0 unwrap unwrap0

# prefix
p = u.punto

install:
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

.PHONY: install
