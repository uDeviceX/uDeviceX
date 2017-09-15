CXXFLAGS = -std=c++11 -Wpedantic -Wall -O3
CXX=g++

S ?= .
CXXFLAGS += -I$S

PROGS = gen_random

all: $(PROGS)

gen_random: gen_random.o common/dump.o

%.o: %.cpp; $(CXX) $(CXXFLAGS) $(OPT) $< -c -o $@
%: %.o; $(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

.PHONY: clean

clean: ; rm -rf *.o $(PROGS)
