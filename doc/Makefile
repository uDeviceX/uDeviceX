ADOC=u.adoc2html

include make/target.mk
all: $(PAGES)
include make/deps.mk

style=-a stylesheet=/udoc/css/main.css
#style=-a stylesheet=/css/main.css

%.html: %.adoc; $(ADOC) $(style) $<
.PHONY: clean

clean:; rm -rf $(PAGES)
