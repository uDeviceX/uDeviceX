CXX=g++
CXXFLAGS= -O3 -Wall -Wpedantic -std=c++11

BIN=$(HOME)/bin/
EXECS = paste

# common
C=../common

BOPDIR = $(HOME)/prefix/bop
LIBS   = -lbop -lbop_serial
LDFLAGS  += -L$(BOPDIR)/lib
CXXFLAGS += -I$(BOPDIR)/include -I$(C)

all: $(EXECS)

install: all
	mkdir -p $(BIN)
	cp $(EXECS) $(BIN)

paste: main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ -lm $(LIBS)

main.o: main.cpp $(C)/macros.h

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(EXECS)

.PHONY: clean all install
