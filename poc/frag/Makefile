include make/ex.mk
include make/frag.mk

# wherer to put headers?

# install target
iF = ../../src/frag.h
D  = ../../doc/frag.md

CXXFLAGS += -O2 -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -I.

# targets F (files), E (examples), T (tests), iD (docs)
# iF (installed F)

install: $F $E $T $D $(iF)

# deps
$F:; main/main .

$E: $F # implicit rule
$T: $F $E; $(run)
$D: doc/frag.md; u.doc doc/frag.md > $D
$(iF): $F; cp $F ../../src/frag.h

.PHONY: clean doc
