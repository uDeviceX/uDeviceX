# install targetn
T=../../src/frag.h

# examples
E=ex/0 ex/1 ex/2 ex/3

f=frag.h
CXXFLAGS += -O2 -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -I.

all: $f $E

$f:; main/main > $@

test: $E; atest ex/0.sh

install: $f; cp $< "$T"

.PHONY: clean test install
clean:; rm -f $f $E
