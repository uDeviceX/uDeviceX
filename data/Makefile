# data directory
D = $(HOME)/.udx/data

# defines O:off, S:sdf, R:rigid, I:[initial condtion]
include make/dep.mk

install: $P
	install0 () {                       \
            f="$$1" d=`dirname "$$f"`;      \
            mkdir -p "$D/$$d" ;             \
	    cp "$$f" "$D/$$d" ;             \
	};                                  \
	for f in $O; do install0 "$$f"; done; \
        for f in $S; do install0 "$$f"; done; \
        for f in $R; do install0 "$$f"; done; \
        for f in $I; do install0 "$$f"; done

.PHONY: install
