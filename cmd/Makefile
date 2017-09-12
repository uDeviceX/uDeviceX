BIN = $(HOME)/bin
B = $(BIN)

P.common = host make/main run/main strtdir argp utest/main u0 u batch/main wrap/main
P.conf =      conf/main conf/make conf/make0 conf/conf conf/run utils
P.daint     = load/daint    make/daint    run/daint    batch/daint    utest/load.daint 
P.leonhard  = load/leonhard make/leonhard run/leonhard batch/leonhard utest/load.leonhard
P.panda     = load/panda    make/panda    run/panda    batch/panda    utest/load.panda
P.acer      = load/acer     make/acer     run/acer     batch/acer     utest/load.acer  

P = $(P.common) $(P.conf) $(P.panda) $(P.daint) $(P.acer) $(P.leonhard)

install: re
install:
	@echo install cmd tools to $B
	@mkdir -p $B
	@for f in $P;                          \
	do t=`tools/transform "$$f"`;          \
           echo "$$f";                         \
           cp "$$f" "$B/$$t";                  \
	done

test:; u.test test/cmd

re:; $(MAKE) -C re BIN=$B

.PHONY: install re test
