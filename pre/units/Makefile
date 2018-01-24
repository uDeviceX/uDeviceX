INST_BIN=~/bin

CXXFLAGS = -std=c++11 -Wpedantic -Wall -O3
CXX=g++

S ?= .
CXXFLAGS += -I$S

PROGS = gen_random

all: $(PROGS)

gen_random: gen_random.o common/dump.o

%.o: %.cpp; $(CXX) $(CXXFLAGS) $(OPT) $< -c -o $@
%: %.o; $(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

P=u.pre
PI=$(PROGS)
install: all
	mkdir -p $(INST_BIN)
	install0 () ( cp $$f $(INST_BIN)/$P.$$f; )    ;  \
	for f in $(PI); do install0; done


.PHONY: clean install all

clean: ; rm -rf *.o $(PROGS)
