# install targetn
T=../../src/frag.h
f=frag.h
CXXFLAGS += -O2 -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -I.

all: $f ex/0

$f:; ./m > $@
ex/0: $f ex/0.cpp
test: ex/0; atest ex/0.sh
install: $f; cp $< "$T"

.PHONY: clean test install
clean:; rm -f $f ex/0
