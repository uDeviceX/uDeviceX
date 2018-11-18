S = $(shell u.cp.s)
include $S/../conf.mk
include $S/../make/example.mk

main: main.o
main.o: main.cpp

install: main; u.install main $(BIN)

clean:; rm -f main main.o
.PHONY: install clean
