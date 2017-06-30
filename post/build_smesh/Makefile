CXX=g++
CXXFLAGS= -O3 -Wall -Wpedantic -std=c++11

BIN=~/bin/
EXECS = po.ss2ply

all: $(EXECS)

install: all
	mkdir -p $(BIN)
	cp $(EXECS) $(BIN)

po.ss2ply: build.o ply.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

build.o: build.cpp ply.h
ply.o: ply.cpp ply.h

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(EXECS)

.PHONY: clean all install
