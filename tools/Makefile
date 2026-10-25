CC = gcc
CFLAGS += -O3 -g -Wall -std=c99 

BIN=$(HOME)/bin

install: build install_cnf install_bops2
install_bops2:; mkdir -p $(BIN) && cp bops2 $(BIN)/bops2vtks && cp bops2 $(BIN)/bops2txts

build: ply2punto

test:
	@$(MAKE) -C ..
	@$(MAKE) -C . build
	@UDX_ROOT="$$PWD/.." sh ../test/mix.contact/2/main

.PHONY: clean all install_cnf install_bops2 install test
clean: ; rm -f ply2punto
