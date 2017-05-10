CXX=g++
CXXFLAGS=-O3 -Wall -Wpedantic -std=c++11

cubic: cubic.cpp gsl_roots.h
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf bbgen
