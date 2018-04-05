CC = c99
CFLAGS += -O3 -g -Wall 

BIN=$(HOME)/bin
 # config directory
CNF=$(HOME)/.udx

EXS=ply2punto ply2off ply.sc ply.sxyz ply.2off off.2cm bop2txt2 u.conf0 \
	u.avgh5 u.avgh5.py u.2davgh5 u.2davgh5.py

install: build install_exs install_cnf install_bops2
install_exs:; mkdir -p $(BIN) && cp $(EXS) $(BIN)
install_cnf:; mkdir -p $(CNF) && cp $(C)   $(CNF)
install_bops2:; mkdir -p $(BIN) && cp bops2 $(BIN)/bops2vtks && cp bops2 $(BIN)/bops2txts

build: ply2punto ply2off

test:; u.test test/*

.PHONY: clean all install_cnf install_exs install_bops2 install test
clean: ; rm -rf ply2punto ply2off
