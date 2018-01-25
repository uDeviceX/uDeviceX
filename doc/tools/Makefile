B = $(HOME)/bin
P = deps index targets all

install:
	@echo install doc tools to $B
	@mkdir -p $B
	@for f in $P;                \
	do t="u.doc.$$f";            \
           cp "$$f" "$B/$$t";        \
	done

.PHONY: clean
