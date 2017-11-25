ADOC=asciidoctor
#ADOC=asciidoc

PAGES = main.html
all: $(PAGES)

args  = -a lext=.html
#args += -a stylesheet=`pwd`/main.cs
args += -a toc=left
args += -a source-highlighter=coderay
args += -a nofooter

%.html: %.adoc
	$(ADOC) $(args) $<

.PHONY: clean

include make/deps.mk

clean:
	rm -rf $(PAGES)
