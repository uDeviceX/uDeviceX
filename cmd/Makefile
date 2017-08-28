BIN = $(HOME)/bin

P.common = u.host make/u.make run/u.run u.strtdir u.argp u.re \
    utest/u.test u.u0 u.u batch/u.batch exec/u.exec
P.conf = conf/u.conf conf/u.conf0 conf/u.conf.make conf/u.conf.make0 conf/u.conf.conf conf/u.conf.run conf/u.conf.re

P.daint  = load/u.load.daint make/u.make.daint run/u.run.daint utest/u.test.load.daint batch/u.batch.daint exec/u.exec.daint
P.panda  = load/u.load.panda make/u.make.panda run/u.run.panda utest/u.test.load.panda batch/u.batch.panda exec/u.exec.panda
P.acer   = load/u.load.acer  make/u.make.acer  run/u.run.acer  utest/u.test.load.acer  batch/u.batch.acer exec/u.exec.acer

P = $(P.common) $(P.conf) $(P.panda) $(P.daint) $(P.acer)

install:
	@echo install cmd tools to $(BIN)
	@mkdir -p $(BIN) && cp $(P) $(BIN)

.PHONY: install
