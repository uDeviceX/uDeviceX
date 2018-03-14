CXX=g++
CXXFLAGS= -O3 -Wall -Wpedantic -std=c++11

BIN=~/bin/
EXECS = po.msd

# common
C=../common

LDFLAGS  += `bop-config --libs`
CXXFLAGS += `bop-config --cflags` 
CXXFLAGS += -I$(C)

all: $(EXECS)

install: all
	mkdir -p $(BIN)
	cp $(EXECS) $(BIN)

po.msd: msd.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm $(LDFLAGS)

msd.o: msd.cpp $(C)/macros.h $(C)/pp_id.h

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(EXECS)

.PHONY: clean all install
