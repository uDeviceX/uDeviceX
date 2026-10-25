CC = gcc
CFLAGS += -O3 -g -Wall -std=c99 

BIN=$(HOME)/bin

EXS=atest/atest atest/ucmp atest/uscale

install: build install_exs install_cnf install_bops2
install_exs:; mkdir -p $(BIN) && cp $(EXS) $(BIN)
install_bops2:; mkdir -p $(BIN) && cp bops2 $(BIN)/bops2vtks && cp bops2 $(BIN)/bops2txts

build: ply2punto

test:
	@$(MAKE) -C .. -f Makefile.concat
	@$(MAKE) -C . build
	@set -e; cd ..; \
		if test -e udx || test -L udx; then mv udx .udx.concat.saved; restore='mv .udx.concat.saved udx'; else restore=':'; fi; \
		if test -e test || test -L test; then mv test .test.concat.saved; restore_test='mv .test.concat.saved test'; else restore_test=':'; fi; \
		trap 'rm -f udx atest.log; eval "$$restore"; eval "$$restore_test"' EXIT INT TERM; \
		ln -s build/concat/udx udx; ln -s src/test test; echo log to atest.log; \
		ATEST_DIR=src/test_data UDX_ROOT="$$PWD" PATH="tools/regression/bin:tools:tools/atest:$$PATH" tools/atest/atest test/mix.contact/2/main > atest.log 2>&1 || { rc=$$?; cat atest.log; exit $$rc; }; cat atest.log

.PHONY: clean all install_cnf install_exs install_bops2 install test
clean: ; rm -f ply2punto
