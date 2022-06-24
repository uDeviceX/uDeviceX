M=../make
include $M/usr.mk
include $M/common.mk
include $M/rules.mk

BOP=../src

LIBS      = -lbop -lbop_serial
LDFLAGS   = -L$(BOP)
INCFLAGS  = -I$(BOP)

PROGS = paste

all: $(PROGS)

test: all
	atest paste.cpp

paste.o: paste.cpp


P=bop
install: $(PROGS)
	mkdir -p $(INST_BIN)
	install0 () ( cp $$f $(INST_BIN)/$P.$$f; )    ;  \
	for f in $(PROGS); do install0; done

clean: ; rm -f $(PROGS) *.o

.PHONY: clean test install all
