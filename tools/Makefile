CC = gcc
CFLAGS += -O3 -g -Wall -std=c99 

BIN=$(HOME)/bin

EXS=atest/atest atest/ucmp atest/uscale

install: build install_exs install_cnf install_bops2
install_exs:; mkdir -p $(BIN) && cp $(EXS) $(BIN)
install_bops2:; mkdir -p $(BIN) && cp bops2 $(BIN)/bops2vtks && cp bops2 $(BIN)/bops2txts

build: ply2punto

test:
	@$(MAKE) -C ..
	@$(MAKE) -C . build
	@set -e; cd ..; \
		if test -e udx || test -L udx; then mv udx .udx.saved; restore='mv .udx.saved udx'; else restore=':'; fi; \
		trap 'rm -f udx atest.log; eval "$$restore"' EXIT INT TERM; \
		ln -s build/udx udx; echo log to atest.log; \
		ATEST_DIR=test_data UDX_ROOT="$$PWD" PATH="tools/regression/bin:tools:tools/atest:$$PATH" tools/atest/atest test/mix.contact/2/main > atest.log 2>&1 || { rc=$$?; cat atest.log; exit $$rc; }; cat atest.log

.PHONY: clean all install_cnf install_exs install_bops2 install test
clean: ; rm -f ply2punto
