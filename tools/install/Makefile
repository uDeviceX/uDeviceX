S = ../..

include $S/conf.mk

B = $(BIN)
P = u
D = install
M = main

include $S/make/tool.mk

test: install; atest test/*

.PHONY: clean test
