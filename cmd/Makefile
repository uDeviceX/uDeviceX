BIN = $(HOME)/bin
B = $(BIN)

P.common = u.host make/u.make run/u.run u.strtdir u.argp \
	    utest/u.test u.u0 u.u batch/u.batch wrap/u.wrap
P.conf = conf/u.conf conf/u.conf0 conf/u.conf.make conf/u.conf.make0 conf/u.conf.conf conf/u.conf.run u.utils

P.daint  = load/u.load.daint make/u.make.daint run/u.run.daint \
	utest/u.test.load.daint batch/u.batch.daint

P.leonhard  = load/u.load.leonhard make/u.make.leonhard run/u.run.leonhard \
	batch/u.batch.leonhard utest/u.test.load.leonhard

P.panda  = load/u.load.panda make/u.make.panda run/u.run.panda \
	 batch/u.batch.panda

P.acer   = load/u.load.acer  make/u.make.acer  run/u.run.acer \
	utest/u.test.load.acer  batch/u.batch.acer

P = $(P.common) $(P.conf) $(P.panda) $(P.daint) $(P.acer) $(P.leonhard)

install: re
install:
	@echo install cmd tools to $B
	@mkdir -p $B && cp $P $B

test:; u.test test/cmd

re:; $(MAKE) -C re BIN=$B

.PHONY: install re test
