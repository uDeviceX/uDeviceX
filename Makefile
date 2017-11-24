C = cmd cmd/argp cmd/par
D = tools cmd post/strt pre/placement pre/units pre/stretch post/ply/cmd post/rbc

install:
	install0 () ( cd "$$d" && make install); \
	for d in $D $C; do install0; done
