CXXFLAGS = -std=c++11 -Wpedantic -Wall -O3
CXX=g++

PROGS = bop2field

#user specific
BOP       = $(HOME)/prefix/bop
LIBS      = -lbop
LDFLAGS   = -L$(BOP)/lib
CXXFLAGS += -I$(BOP)/include

all: $(PROGS)
%: %.o; $(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

bop2field: bop2field.o
bop2field.o: bop2field.cpp

clean:
	rm -f $(PROGS) *.o

.PHONY: clean
