CXXFLAGS+=-std=c++11 -O3 -Wall

BIN=$(HOME)/bin
CNF=$(HOME)/.udx # config directory

EXS=argp argp.awk ply2punto ply.sc ply.sxyz ply.2off off.2cm
C=u.sh # configs

all: ply2punto

install: all install_exs install_cnf
install_exs:; mkdir -p $(BIN) && cp $(EXS) $(BIN)
install_cnf:; mkdir -p $(CNF) && cp $(C)   $(CNF)

ply2punto: ply2punto.cpp

.PHONY: clean all install_cnf install_exs install
clean: ; rm -rf ply2punto
