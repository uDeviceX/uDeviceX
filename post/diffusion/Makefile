CXX=g++
CXXFLAGS= -O3 -Wall -Wpedantic -std=c++11

BIN=~/bin/
EXECS = po.diffCoeff

BOPDIR = $(HOME)/prefix/bop
LIBS = -lbop
LDFLAGS += -L$(BOPDIR)/lib
CXXFLAGS += -I$(BOPDIR)/include 

all: $(EXECS)

install: all
	mkdir -p $(BIN)
	cp $(EXECS) $(BIN)

po.diffCoeff: diffusion.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ -lm $(LIBS)

diffusion.o: diffusion.cpp

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(EXECS)

.PHONY: clean all install
