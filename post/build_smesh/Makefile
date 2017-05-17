CXX=g++
CXXFLAGS= -O3 -Wall -Wpedantic -std=c++11

EXECS = build

default: all
all: $(EXECS)

build: build.o ply.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

build.o: build.cpp ply.h
ply.o: ply.cpp ply.h

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(EXECS)
