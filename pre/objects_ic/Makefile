CXX      = g++
CXXFLAGS = -O3 -Wall -Wpedantic
LDFLAGS  =

all: main

main: main.o particles.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

main.o: main.cpp type.h utils.h
particles.o: particles.cpp type.h utils.h

clean:; rm -rf main *.o
.PHONY: clean
