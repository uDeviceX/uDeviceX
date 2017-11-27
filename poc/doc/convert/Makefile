D = adoc2html adoc2md md2adoc

install:
	install0 () ( cd "$$d" && make install); \
	for d in $D; do install0; done

test:; atest test/*

.PHONY: install test
