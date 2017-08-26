# install target
iF = ../../src/frag.h
D  = ../../doc/frag.md

CXXFLAGS += -O2 -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -I.


# targets F (files), E (examples), test, D (docs)
# iF (installed F)

E = ex/0 ex/1 ex/2 ex/3 ex/4
F = frag.h

# rule
run = for f in $E; do $$f; done && \
	atest ex/0.sh

install: $F $E $T $D $(iF)

# deps
$F: main/main0 main/main
$F:; main/main .

$E: $F # implicit rule
test: $F $E; $(run)
$D: doc/frag.md; u.doc doc/frag.md > $D
$(iF): $F; cp $F ../../src/frag.h

.PHONY: clean doc

clean:; rm -f $E $F
