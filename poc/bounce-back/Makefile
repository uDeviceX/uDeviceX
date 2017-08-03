CXX=g++
CXXFLAGS=-O3 -Wall -Wpedantic -std=c++11

all: bbgen inertia

bbgen: bbgen.cpp bb.h roots.h
	$(CXX) $(CXXFLAGS) $< -o $@

inertia: inertia.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf bbgen inertia
