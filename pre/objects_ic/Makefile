CXX      = g++
CXXFLAGS = -O3 -Wall -Wpedantic
LDFLAGS  =

BIN=$(HOME)/bin

all: main

main: main.o particles.o matrices.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

main.o: main.cpp type.h utils.h particles.h matrices.h
particles.o: particles.cpp type.h utils.h
matrices.o: matrices.cpp type.h

clean:; rm -rf main *.o
.PHONY: clean install

install: main
	mkdir -p $(BIN)
	cp main $(BIN)/u.obj-ic
