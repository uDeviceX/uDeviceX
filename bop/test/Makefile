M=../make

include $M/usr.mk
include $M/common.mk
include $M/rules.mk

BOP=$(INST_LIB)

INCFLAGS = -I$(BOP)/include
LIBS_B   = -L$(BOP)/lib -lbop

LIBS     = $(LIBS_B) -lbop_serial
LIBS_MPI = $(LIBS_B) -lbop_mpi

PROGS = read write write_mpi

all: $(PROGS)

read.o: read.cpp check.h
write.o: write.cpp check.h

$(MPI_TOK)write_mpi.o: write_mpi.cpp check.h

test: all; atest *.cpp

clean:; rm -rf $(PROGS) *.o

.PHONY: all clean test
