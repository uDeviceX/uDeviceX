S = ../..

include $S/conf.mk

B = $(BIN)
P = u
D = pkg-config
M = main
SED = sed

include $S/make/tool.mk

main: main.in
	$(SED) "s,@PKG@,$(PKG),g" $< > $@; \
	chmod +x $@

test: install

.PHONY: clean test
