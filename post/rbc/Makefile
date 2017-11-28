D = lib cmd

install:
	install0 () ( cd "$$d" && make install); \
	for d in $D; do install0; done

test:; atest test/*

.PHONY: install test
