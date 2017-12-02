ADOC=u.adoc2html

include make/target.mk
all: $(PAGES)
include make/deps.mk

style=-a stylesheet=adoc.css

%.html: %.adoc; $(ADOC) $(style) $<
.PHONY: clean

clean:; rm -rf $(PAGES)
