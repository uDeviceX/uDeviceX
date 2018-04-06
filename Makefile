A = cmd cmd/argp cmd/par cmd/u cmd/build cmd/case poc/doc/convert/adoc2html poc/ply2vtk post/ply post/field cmd/utest
B = tools cmd post/punto post/strt post/wall pre/placement pre/units pre/stretch
C = post/ply/cmd post/rbc post/data conf data cmd/cp cmd/scatter cmd/var cmd/edb

install:
	install0 () ( cd "$$d" && make install); \
	for d in $A $B $C; do install0; done

.PHONY: install
