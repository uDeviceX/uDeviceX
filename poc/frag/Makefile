# install targetn
T=../../src/frag.h

all: frag.h ex0

frag.h:; ./m > $@
ex0: ex0.cpp
test: ex0; atest ex
install: frag.h; cp $< "$T"

.PHONY: clean test install
clean:; rm -f frag.h ex0
