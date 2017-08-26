# install targetn
T=../../src/frag.h
f=frag.h
CXXFLAGS += -O2 -g
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -I.

all: $f ex/ex0

$f:; ./m > $@
ex/ex0: $f ex/ex0.cpp
test: ex/ex0; atest ex/main
install: $f; cp $< "$T"

.PHONY: clean test install
clean:; rm -f $f ex/ex0
