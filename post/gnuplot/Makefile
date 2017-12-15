D = cmd

install:
	install0 () ( cd "$$d" && make install); \
	for d in $D $C; do install0; done
