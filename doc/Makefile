ADOC=asciidoctor
#ADOC=asciidoc

include make/target.mk

all: $(PAGES)

args  = -a lext=.html
#args += -a stylesheet=`pwd`/main.cs
args += -a toc=left
args += -a source-highlighter=coderay
args += -a nofooter

%.html: %.adoc
	$(ADOC) $(args) $<

.PHONY: clean


clean:
	rm -rf $(PAGES)
