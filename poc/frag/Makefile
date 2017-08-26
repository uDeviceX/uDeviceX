# install targetn
T=../../src/frag.h

# examples
E=ex/0 ex/1 ex/2 ex/3

f=frag.h
CXXFLAGS += -O2 -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -I.

all: $f $E
$E: frag.h

$f:; main/main > $@

test: $E
	atest ex/0.sh && \
	for e in $E; do $$e; done

install: $f; cp $< "$T"

.PHONY: clean test install
clean:; rm -f $f $E
