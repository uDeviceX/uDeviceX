BIN = $(HOME)/bin
# config
C   = $(HOME)/.udx
B = $(BIN)

# path to `src`
S = ../../src

P = ic ic0 rbc rbc0 rig rig0 s sdf sdf0 util

# prefix
p = u.cp

install:
	mkdir -p "$B".
	install0 () ( cp $$f $B/$p.$$f )    ;  \
	for f in $P; do install0; done      ;  \
	s () ( cd "$S"; pwd )               ;  \
	s > "$C/s"

.PHONY: install
