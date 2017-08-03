CXX = gcc
CXXFLAGS = -O3 -Wall -Wpedantic

all: deltas

deltas: deltas.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ 

clean:
	rm -rf deltas
