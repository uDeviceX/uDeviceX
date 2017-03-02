#!/usr/bin/awk -f

# Convert uDeviceX old format to off [1, 2]
#
# OFF numVertices numFaces numEdges
# x y z
# x y z
# ... numVertices like above
# NVertices v1 v2 v3 ... vN
# MVertices v1 v2 v3 ... vM
# ... numFaces like above

# [1] https://en.wikipedia.org/wiki/OFF_(file_format)
# [2] http://shape.cs.princeton.edu/benchmark/documentation/off_format.html

# TEST: ud2off.t1
# ./ud2off.awk test_data/rbc.dat > rbc.out.off

function nl() { # next line
    getline < fn
}

function emptyp() {
    return $0 ~ /^[ \t]*$/
}

function init() {
    fn = ARGC < 2 ? "-" : ARGV[1]
}

function read_vert(   ib, iv) {
    iv = 0
    while (getline < fn > 0) if ($0 ~ /Atoms/) break
    nl() # skip empty line
    while (getline < fn > 0) { # vertices
	if (emptyp()) break
	ib = 4; xx[iv] = $(ib++); yy[iv] = $(ib++); zz[iv] = $(ib++); iv++
    }
    nv = iv
}

function read_faces(  ib, ifa) {
    ifa = 0
    while (getline < fn > 0) if ($0 ~ /Angles/) break
    nl() # empty line
    while (getline < fn > 0) { # faces
	if (emptyp()) break
	ib = 3; f0[ifa] = $(ib++); f1[ifa] = $(ib++); f2[ifa] = $(ib++); ifa++
    }
    nf = ifa
}

function write_header() {
    print "OFF"
    print nv, nf, ne
}

function write_vert(   iv) {
    for (iv = 0; iv < nv; iv++)
	print xx[iv], yy[iv], zz[iv]
}

function write_faces(   ifa) {
    nv_per_face = 3
    for (ifa = 0; ifa < nf; ifa++)
	print nv_per_face, f0[ifa], f1[ifa], f2[ifa]
}

BEGIN {
    # input file name
    init()
    read_vert()
    read_faces()
    ne = 0 # no edges

    write_header()
    write_vert()
    write_faces()
}
