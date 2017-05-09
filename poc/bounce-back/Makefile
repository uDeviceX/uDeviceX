CXX=g++
CXXFLAGS=-O3 -Wall -Wpedantic -std=c++11

bbgen: bbgen.cpp bb.h roots.h
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf bbgen
