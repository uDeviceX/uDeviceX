CXX=g++
CXXFLAGS = -Wpedantic -Wall -Wextra -O3 -std=c++11

%.o: %.cpp; $(CXX) $(CXXFLAGS) $(OPT) -c $< -o $@
%: %.o; $(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)


INST_BIN=~/bin

CXXFLAGS += `bov-config --cflags`
LIBS     += `bov-config --libs`
CXXFLAGS += `bop-config --cflags`
LIBS     += `bop-config --libs`

PROGS=color
PI=$(PROGS)
P=u.p2m

all: $(PROGS)

color.o: color.cpp

.PHONY: clean install test
clean:; rm -rf *.o $(PROGS)

install: $(PROGS)
	mkdir -p $(INST_BIN)
	install0 () ( cp $$f $(INST_BIN)/$P.$$f; )    ;  \
	for f in $(PI); do install0; done

test: install
	atest color.cpp
