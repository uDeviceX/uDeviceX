A = conf tools/install tools/pkgconfig cmd cmd/argp cmd/par cmd/u poc/doc/convert/adoc2html poc/ply2vtk post/ply post/field cmd/utest
B = tools cmd post/punto post/strt post/wall pre/placement pre/units pre/stretch pre/geomview
C = post/ply/cmd post/rbc post/data data cmd/cp cmd/scatter cmd/var cmd/edb cmd/maxima
D = lib/ply/src pkgconfig/ply pre/uconf/src cmd/python

install:
	install0 () ( cd "$$d" && make install); \
	for d in $A $B $C $D; do install0; done

.PHONY: install
