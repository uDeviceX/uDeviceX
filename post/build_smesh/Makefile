CXX=g++
CXXFLAGS= -O3 -Wall -Wpedantic -std=c++11

EXECS = rw build

default: all
all: $(EXECS)

rw: rw.o ply.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

build: build.o ply.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(EXECS)
