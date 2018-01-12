# converter
ADOC ?= u.adoc2html 
# root on server
ROOT ?= ""

include make/target.mk
all: $(PAGES)
include make/deps.mk

style=-a stylesheet=$(ROOT)/css/main.css -a coderay-css=style
#style=-a stylesheet=/css/main.css

%.html: %.adoc; $(ADOC) $(style) $<
.PHONY: clean

clean:; rm -rf $(PAGES)
