CXX=g++
CXXFLAGS = -Wpedantic -Wall -Wextra -O3

%.o: %.cpp; $(CXX) $(CXXFLAGS) $(OPT) -c $< -o $@
%: %.o; $(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)


INST_BIN=~/bin
BOV=$(HOME)/prefix/bov
BOP=$(HOME)/prefix/bop

CXXFLAGS += -I$(BOV)/include
LIBS     += -L$(BOV)/lib -lbov -lbov_serial
CXXFLAGS += -I$(BOP)/include
LIBS     += -L$(BOP)/lib -lbop

PROGS=avg
P=u

all: $(PROGS)

avg.o: avg.cpp

.PHONY: clean install test
clean:; rm -rf $(PROGS)

install:
	mkdir -p $(INST_BIN)
	install0 () ( cp $$f $(INST_BIN)/$P.$$f; )    ;  \
	for f in $(P); do install0; done

test:
	atest avg.cpp
