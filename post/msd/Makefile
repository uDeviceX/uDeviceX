CXX=g++
CXXFLAGS= -O3 -Wall -Wpedantic -std=c++11

BIN=~/bin/
EXECS = po.msd

# common
C=../common

BOPDIR = $(HOME)/prefix/bop
LIBS = -lbop
LDFLAGS += -L$(BOPDIR)/lib
CXXFLAGS += -I$(BOPDIR)/include -I$(C)

all: $(EXECS)

install: all
	mkdir -p $(BIN)
	cp $(EXECS) $(BIN)

po.msd: msd.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ -lm $(LIBS)

msd.o: msd.cpp $(C)/macros.h $(C)/pp_id.h

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(EXECS)

.PHONY: clean all install
