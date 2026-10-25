CC = gcc
CFLAGS += -O3 -g -Wall -std=c99 

build: ply2punto

test:
	@$(MAKE) -C ..
	@$(MAKE) -C . build
	@UDX_ROOT="$$PWD/.." sh ../test/mix.contact/2/main

.PHONY: clean all build test
clean: ; rm -f ply2punto
