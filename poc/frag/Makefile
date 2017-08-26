# install target
T=../../src/frag
f=frag/ncell.h

# examples
E=ex/0 ex/1 ex/2 ex/3

CXXFLAGS += -O2 -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -I.

all: $f $E
$E:  $f

$f:; main/main > $@

test: $E
	atest ex/0.sh && \
	for e in $E; do $$e; done

install: $f
	mkdir -p  $T
	cp frag.h $T

.PHONY: clean test install
clean:; rm -f $f $E
