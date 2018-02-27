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

PROGS=binning
PI=$(PROGS)
P=u

all: $(PROGS)

binning.o: binning.cpp

.PHONY: clean install test
clean:; rm -rf *.o $(PROGS)

install: $(PROGS)
	mkdir -p $(INST_BIN)
	install0 () ( cp $$f $(INST_BIN)/$P.$$f; )    ;  \
	for f in $(PI); do install0; done

test: install
	atest binning.cpp
