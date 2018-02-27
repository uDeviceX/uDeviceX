CXX=g++
CXXFLAGS = -Wpedantic -Wall -Wextra -O3 -std=c++11

%.o: %.cpp; $(CXX) $(CXXFLAGS) $(OPT) -c $< -o $@
%: %.o; $(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)


INST_BIN=~/bin
BOV=$(HOME)/prefix/bov
BOP=$(HOME)/prefix/bop

CXXFLAGS += -I$(BOV)/include
LIBS     += -L$(BOV)/lib -lbov -lbov_serial
CXXFLAGS += -I$(BOP)/include
LIBS     += -L$(BOP)/lib -lbop -lbop_serial

PROGS=color.density color.minmax
PI=$(PROGS) bops2colordensity colors2minmaxs
P=u

all: $(PROGS)

color.density.o: color.density.cpp
color.minmax.o: color.minmax.cpp


.PHONY: clean install test
clean:; rm -rf $(PROGS)

install:
	mkdir -p $(INST_BIN)
	install0 () ( cp $$f $(INST_BIN)/$P.$$f; )    ;  \
	for f in $(PI); do install0; done

test:
	atest color.density.cpp
