# install targetn
T=../../src/frag.h
f=frag.h

all: $f ex0

$f:; ./m > $@
ex0: $f ex0.cpp
test: ex0; atest ex
install: $f; cp $< "$T"

.PHONY: clean test install
clean:; rm -f $f ex0
