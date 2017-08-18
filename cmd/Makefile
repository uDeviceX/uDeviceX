BIN = $(HOME)/bin

P.common = u.host make/u.make run/u.run u.strtdir u.argp \
    utest/u.test
P.conf = conf/u.conf conf/u.conf.make conf/u.conf.make0 conf/u.conf.conf conf/u.conf.run 

P.daint  = make/u.load.daint make/u.make.daint run/u.run.daint utest/u.test.load.daint
P.panda  = make/u.load.panda make/u.make.panda run/u.run.panda utest/u.test.load.panda
P.acer   = make/u.load.acer  make/u.make.acer  run/u.run.acer  utest/u.test.load.acer

P = $(P.common) $(P.conf) $(P.panda) $(P.daint) $(P.acer)

install:; mkdir -p $(BIN) && cp $(P) $(BIN)

.PHONY: install
