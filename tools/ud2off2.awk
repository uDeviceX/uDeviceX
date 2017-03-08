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

function read_header() {
    getline < fn; nv = $1
    getline < fn; ne = $1
    getline < fn; nf = $1
    nl(); nl() # skip header
}

function read_vert(   ib, iv) {
    for (iv = 0; iv < nv; iv++) { # vertices
        getline < fn
        ib = 4; xx[iv] = $(ib++); yy[iv] = $(ib++); zz[iv] = $(ib++)
    }
    nl()
}

function skip_edges(  ie) {
    for (ie = 0; ie < ne; ie++) nl()
    nl()
}

function read_faces(  ib, ifa) {
    for (ifa = 0; ifa < nf; ifa++) { # faces
        getline < fn
        ib = 3; ff0[ifa] = $(ib++); ff1[ifa] = $(ib++); ff2[ifa] = $(ib++)
    }
    nl()
}

function o2z(i) { return i-1 }

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
    for (ifa = 0; ifa < nf; ifa++) {
        f0 = ff0[ifa]; f1 = ff1[ifa]; f2 = ff2[ifa]
        print nv_per_face, o2z(f0), o2z(f1), o2z(f2)
    }
}

BEGIN {
    # input file name
    init()
    read_header()
    read_vert()
    skip_edges()
    read_faces()
    ne = 0 # no edges

    write_header()
    write_vert()
    write_faces()
}
