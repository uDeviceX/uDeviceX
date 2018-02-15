# converter
ADOC = u.adoc2html
# root on server
ROOT =

include make/target.mk
all: $(PAGES)
include make/deps.mk

style=-a stylesheet="$(ROOT)/css/main.css"  \
      -a pygments-style=colorful            \
      -a toc=preamble                       \
      -a sectnums

all.html: all.adoc; $(ADOC) $(style) -a src=`u.cp.s` $<
%.html: %.adoc; $(ADOC) $(style) $<
.PHONY: clean

clean:; rm -rf $(PAGES)
