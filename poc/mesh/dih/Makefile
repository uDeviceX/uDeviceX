GNUPLOT=gnuplot
main.png: main; $(GNUPLOT) $<
main.png: inc/label data/i

inc/label:; cmd/label data/verts data/offset > $@
data/i:; cmd/line data/verts data/line > $@


.PHONY: clean

clean:; rm -f data/i inc/label main.png
