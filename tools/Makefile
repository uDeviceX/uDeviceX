CC = gcc
CFLAGS += -O3 -g -Wall -std=c99 

BIN=$(HOME)/bin

EXS=atest/uscale

install: build install_exs install_cnf install_bops2
install_exs:; mkdir -p $(BIN) && cp $(EXS) $(BIN)
install_bops2:; mkdir -p $(BIN) && cp bops2 $(BIN)/bops2vtks && cp bops2 $(BIN)/bops2txts

build: ply2punto

test:
	@$(MAKE) -C ..
	@$(MAKE) -C . build
	@UDX_ROOT="$$PWD/.." sh ../test/mix.contact/2/main

.PHONY: clean all install_cnf install_exs install_bops2 install test
clean: ; rm -f ply2punto
