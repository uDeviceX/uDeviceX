BIN = $(HOME)/bin
B = $(BIN)

P.common = host make/main robin run/main strtdir batch/main wrap/main
P.conf = conf/main conf/make conf/make0 conf/conf conf/run utils
P.daint     = load/daint    make/daint    run/daint    batch/daint    post.load/daint
P.leonhard  = load/leonhard make/leonhard run/leonhard batch/leonhard post.load/leonhard
P.panda     = load/panda    make/panda    run/panda    batch/panda    post.load/panda
P.acer      = load/acer     make/acer     run/acer     batch/acer     post.load/acer

P = $(P.common) $(P.conf) $(P.panda) $(P.daint) $(P.acer) $(P.leonhard)

install:
	@echo install cmd tools to $B
	@mkdir -p $B
	@for f in $P;                          \
	do t=`tools/transform "$$f"`;          \
           cp "$$f" "$B/$$t";                  \
	done

test:; atest test/*

re:; $(MAKE) -C re BIN=$B

.PHONY: install re test
