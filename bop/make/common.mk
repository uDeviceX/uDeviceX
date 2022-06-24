# where to install tools
INST_BIN = $(HOME)/bin
# where to install library
INST_LIB = $(prefix)/bop

CXXFLAGS    = -std=c++11 -Wpedantic -Wall -O3 -fPIC
CXX         = g++
MPICXX      = mpic++
MPICXXFLAGS = $(CXXFLAGS)
