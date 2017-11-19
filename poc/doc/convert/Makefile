d = doc
B = $(HOME)/bin

p = u
P = adoc2md adoc2md0

install:
	mkdir -p "$B"
	install0 () ( cp $$f $B/$p.$$f; )    ;  \
	for f in $P; do install0; done

.PHONY: install
