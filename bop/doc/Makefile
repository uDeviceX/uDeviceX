# converter
ADOC = asciidoctor
# root on server
ROOT =

# src of the bop
S=../src

a  = -a lext=.html
a += -a source-highlighter=pygments
a += -a pygments-style=colorful
a += -a icons=font
a += -a nofooter
a += -a disable-javascript
a += -a linkcss

include make/target.mk
all: $(PAGES)
include make/deps.mk

style=-a stylesheet="$(ROOT)/css/main.css"  \
      -a include="$(ROOT)/include"          \
      -a toc=left                           \
      -a sectnums                           \
      -a src=$S

%.html: %.adoc; $(ADOC) $a -b html5 $(style) $<
.PHONY: clean

clean:; rm -rf $(PAGES)
