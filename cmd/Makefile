BIN = $(HOME)/bin

P.common = u.host u.make u.run u.conf u.conf.make u.conf.make0 u.conf.conf u.conf.run u.strtdir u.argp \
    utest/u.test
P.daint  = u.load.daint u.run.daint utest/u.test.load.daint
P.panda  = u.load.panda u.run.panda utest/u.test.load.panda
P.acer   = u.load.acer  u.run.acer  utest/u.test.load.acer

P = $(P.common) $(P.panda) $(P.daint) $(P.acer)

install:; mkdir -p $(BIN) && cp $(P) $(BIN)

.PHONY: install
