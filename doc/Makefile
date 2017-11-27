ADOC=u.adoc2html

include make/target.mk
all: $(PAGES)
include make/deps.mk

%.html: %.adoc; $(ADOC) $<
.PHONY: clean

clean:; rm -rf $(PAGES)
