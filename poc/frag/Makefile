# wherer to put headers?
d = frag

# install target
T = ../../src/$d
F = $d/common.h $d/fro.h $d/to.h

# examples
E = ex/0 ex/1 ex/2 ex/3

CXXFLAGS += -O2 -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -I.

all: $F $E
$E:  $F

$F:
	main/main $d

test: $E
	atest ex/0.sh && \
	for e in $E; do $$e; done

install: $F
	mkdir -p  $T
	cp frag.h $T

.PHONY: clean test install
clean:; rm -f $F $E
