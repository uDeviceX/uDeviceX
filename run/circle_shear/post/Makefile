BIN = $(HOME)/bin # where to install
CXXFLAGS = -std=c++11 -Wpedantic -Wall -O3
CXX=g++

PROGS = bop2field

#user specific
BOP       = /home/amlucas/bop/bop
LIBS      = -lbop
LDFLAGS   = -L$(BOP)
CXXFLAGS += -I$(BOP)

all: $(PROGS)
%: %.o; $(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

bop2field: bop2field.o libbop
bop2field.o: bop2field.cpp

libbop:; make -C $(BOP)

clean:
	rm -f $(PROGS) *.o
	make -C $(BOP) clean

.PHONY: clean install libbop
