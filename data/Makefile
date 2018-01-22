# data directory
D = $(HOME)/.udx/data

include make/dep.mk

install: $P
	install0 () {                       \
            f="$$1";                        \
            d=`dirname "$$f"`;              \
            mkdir -p "$D/$$d" ;             \
	    cp "$$f" "$D/$$d" ;             \
	};                                  \
	for f in $P; do install0 "$$f"; done

.PHONY: install
