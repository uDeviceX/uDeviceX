M=../make
BOP=../src

include $M/usr.mk
include $M/common.mk
include $M/rules.mk

LIBS      = -lbop -lbop_serial
LDFLAGS   = -L$(BOP)
INCFLAGS  = -I$(BOP)

PROGS = bop2vtk bop2txt

all: $(PROGS)

test: all
	atest *.cpp

bop2vtk.o: bop2vtk.cpp check.h
bop2txt.o: bop2txt.cpp check.h

install:
	mkdir -p $(INST_BIN)
	cp $(PROGS) $(INST_BIN)

clean:
	rm -f $(PROGS) *.o

.PHONY: test clean install
