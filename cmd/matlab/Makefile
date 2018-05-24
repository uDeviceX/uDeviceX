S = ../..
P = u
D = matlab
M = main util

PREFIX = $(HOME)
B = $(HOME)/bin

include $S/make/tool.mk
main: main.in
	./replace_matlab main.in > main
	chmod +x main
.PHONY: install clean
clean:; rm -f main
