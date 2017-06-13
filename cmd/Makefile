BIN = $(HOME)/bin

P.common = u.host u.make u.run
P.daint  = u.load.daint u.run.daint
P.panda  = u.load.panda u.run.panda
P.acer   = u.load.acer  u.run.acer

P = $(P.common) $(P.panda) $(P.daint) $(P.acer)

install:; mkdir -p $(BIN) && cp $(P) $(BIN)

.PHONY: install
