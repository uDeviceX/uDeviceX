# converter
ADOC = u.adoc2html
# root on server
ROOT =

include make/target.mk
all: $(PAGES)
include make/deps.mk

style=  \
      -a include="$(ROOT)/include"          \
      -a toc=left                           \
      -a sectnums

%.html: %.adoc; $(ADOC) $(style) -a src=`u.cp.s` $<
.PHONY: clean

clean:; rm -rf $(PAGES)
