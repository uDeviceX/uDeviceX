CXX=g++
CXXFLAGS= -Wall -Wpedantic -O3 -std=c++11

default: test_ply test_inmesh

test_ply: test_ply.cpp mesh.h
	$(CXX) $(CXXFLAGS) $< -o $@

test_inmesh: test_inmesh.cpp mesh.h collision.h
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf test_ply test_inmesh
