S = ../..
P = u
D = matlab
M = main util

PREFIX = $(HOME)
B = $(PREFIX)/bin
MATLABPATH=$(PREFIX)/lib/matlab

include $S/make/tool.mk
main: main.in
	mkdir -p "$(MATLABPATH)"
	MATLABPATH="$(MATLABPATH)" ./replace_matlab main.in > main
	chmod +x main

.PHONY: install clean
clean:; rm -f main
