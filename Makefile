D = tools cmd post/strt pre/placement post/ply/cmd post/rbc

install:
	install0 () ( cd "$$d" && make install); \
	for d in $D; do install0; done
