M=../make

include $M/usr.mk
include $M/common.mk
include $M/rules.mk

BOP=../src

LIBS      = -lbop -lbop_serial
LDFLAGS   = -L$(BOP)
INCFLAGS  = -I$(BOP)

PROGS = seq2bop rwascii

all: $(PROGS)

test: all
	atest rwascii.cpp

seq2bop.o: seq2bop.cpp
rwascii.o: rwascii.cpp

install:
	mkdir -p $(INST_BIN)
	cp $(PROGS) $(INST_BIN)

clean: ; rm -f $(PROGS) *.o

.PHONY: clean test
