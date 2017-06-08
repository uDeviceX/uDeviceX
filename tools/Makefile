CXXFLAGS+=-std=c++11 -O3 -Wall

BIN=$(HOME)/bin
EXS=argp argp.awk ply2punto ply.sc ply.sxyz

all: ply2punto

install: all
	cp $(EXS) $(BIN)

ply2punto: ply2punto.cpp

.PHONY: clean
clean: ; rm -rf ply2punto
