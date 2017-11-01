CXX=g++
CXXFLAGS = -Wpedantic -Wall -Wextra -O3

%.o: %.c; $(CXX) $(CXXFLAGS) $(OPT) -c $< -o $@
%: %.o; $(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)


INST_BIN=~/bin
BOV=$(HOME)/prefix/bov/
BOP=$(HOME)/prefix/bop/

CXXFLAGS += -I$(BOV)/include
LIBS     += -L$(BOV)/lib -lbov
CXXFLAGS += -I$(BOP)/include
LIBS     += -L$(BOP)/lib -lbop

PROGS=color.density
P=u

all: $(PROGS)

color.density.o: color.density.cpp

.PHONY: clean install
clean:; rm -rf $(PROGS)

install:
	mkdir -p $(INST_BIN)
	install0 () ( cp $$f $(INST_BIN)/$P.$$f; )    ;  \
	for f in $(PROGS); do install0; done

