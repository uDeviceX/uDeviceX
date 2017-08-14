BIN = $(HOME)/bin

P.common = u.host make/u.make u.run u.conf u.conf.make u.conf.make0 u.conf.conf u.conf.run u.strtdir u.argp \
    utest/u.test
P.daint  = make/u.load.daint make/u.make.daint u.run.daint utest/u.test.load.daint
P.panda  = make/u.load.panda make/u.make.panda u.run.panda utest/u.test.load.panda
P.acer   = make/u.load.acer  make/u.make.acer  u.run.acer  utest/u.test.load.acer

P = $(P.common) $(P.panda) $(P.daint) $(P.acer)

install:; mkdir -p $(BIN) && cp $(P) $(BIN)

.PHONY: install
