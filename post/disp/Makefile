CXX=g++
CXXFLAGS= -O3 -Wall -Wpedantic -std=c++11

BIN=~/bin/
EXECS = po.disp

# common
C=../common

LDFLAGS  += `bop-config --libs`
CXXFLAGS += `bop-config --cflags`
CXXFLAGS += -I$(C)

all: $(EXECS)

install: all
	mkdir -p $(BIN)
	cp $(EXECS) $(BIN)

po.disp: main.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -lm $(LDFLAGS)

main.o: main.cpp $(C)/macros.h $(C)/pp_id.h

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(EXECS)

test:
	atest main.cpp

.PHONY: clean all install test
