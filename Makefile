M = \
conf \
tools/install \
tools/pkgconfig \
cmd \
cmd/argp \
cmd/par \
cmd/u \
poc/doc/convert/adoc2html \
poc/ply2vtk \
post/ply \
post/field \
cmd/utest \
tools \
post/punto \
post/strt \
post/wall \
pre/placement \
pre/units \
pre/stretch \
pre/geomview \
post/ply/cmd \
post/rbc \
post/data \
data \
cmd/cp \
cmd/scatter \
cmd/var \
cmd/edb \
cmd/maxima \
lib/ply/src \
pkgconfig/ply \
pre/uconf/src \
cmd/python \


install:
	for d in $M; do ( cd "$$d" && make install) || exit 2; done
