C = cmd cmd/argp
D = tools cmd post/strt pre/placement pre/units post/ply/cmd post/rbc

install:
	install0 () ( cd "$$d" && make install); \
	for d in $D $C; do install0; done
