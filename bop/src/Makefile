M=../make
include $M/usr.mk
include $M/common.mk
include $M/rules.mk

inc=bop_common.h bop_serial.h bop_mpi.h
lib=libbop.a libbop_serial.a libbop_mpi.a

all: $(lib)

libbop.a: bop_common.o utils.o header.o
	ar rcs $@ $^

libbop_serial.a: bop_serial.o header.o
	ar rcs $@ $^

libbop_mpi.a: $(MPI_TOK)bop_mpi.o header.o
	ar rcs $@ $^

bop_common.o: bop_common.cpp bop_common.h macros.h utils.h type.h
utils.o:  utils.cpp utils.h bop_common.h type.h
header.o: header.cpp header.h bop_common.h utils.h macros.h type.h

bop_serial.o: bop_serial.cpp bop_serial.h macros.h utils.h header.h type.h
$(MPI_TOK)bop_mpi.o: bop_mpi.cpp bop_mpi.h macros.h utils.h header.h type.h

.PHONY: clean install
clean:; rm -rf *.o *.a

install:
	mkdir -p $(INST_BIN)
	mkdir -p $(INST_LIB)/include
	mkdir -p $(INST_LIB)/lib
	cp $(inc) $(INST_LIB)/include/
	cp $(lib) $(INST_LIB)/lib/

