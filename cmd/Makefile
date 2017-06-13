BIN = $(HOME)/bin

P.common = u.host u.make
P.daint  = u.load.daint
P.panda  = u.load.panda
P.acer   = u.load.acer

P = $(P.common) $(P.panda) $(P.daint) $(P.acer)

install:; mkdir -p $(BIN) && cp $(P) $(BIN)

.PHONY: install
