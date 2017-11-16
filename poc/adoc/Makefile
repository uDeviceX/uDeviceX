m=main
a=--backend=xhtml11
$m.html: $m.adoc main.conf
	asciidoc -f main.conf $a $<

.PHONY: clean

clean:; rm -f $m.html

