#!/usr/bin/awk -f

# convert
# x1 y1 z1  x2 y2 z2
# to vector to vector field in vtk
# x2 - x1, y2 - y1, z2 - z1

function file_version() {
    printf "# vtk DataFile Version 2.0\n"
}

function header() {
    printf "Created with uC\n"
}

function format() {
    printf "ASCII\n"
}

function structure() {
    polydata()
}

function polydata(   dataType, size, m, iv) {
    dataType = "float"
    printf "DATASET POLYDATA\n"
    printf "POINTS %d %s\n", nv, dataType
    for (iv=0; iv<nv; iv++)
	printf "%g %g %g\n",  xx[iv], yy[iv], zz[iv]
}

function point_attributes(    dataType, dataName, iv) {
    dataType = "float"
    dataName = "F"
    printf "POINT_DATA %d\n", nv
    printf "VECTORS %s %s\n", dataName, dataType
    for (iv=0; iv<nv; iv++) {
	printf "%s %s %s\n", vvx[iv], vvy[iv], vvz[iv]
    }
}

function attributes() {
    point_attributes()
}

{ # read file
    np = NR - 1; i = 1
    xx1[np] = $(i++); yy1[np] = $(i++); zz1[np] = $(i++)
    xx2[np] = $(i++); yy2[np] = $(i++); zz2[np] = $(i++)
}

function process(   iv, ip) {
    for (ip = iv = 0; ip < np; ip++) {
	xx[iv] = xx1[ip]; yy[iv] = yy1[ip]; zz[iv] = zz1[ip]; iv++
    }
    nv = iv
    
    for (iv = 0; iv < nv; iv++) {
	vvx[iv] = xx2[iv] - xx1[iv];
	vvy[iv] = yy2[iv] - yy1[iv];
	vvz[iv] = zz2[iv] - zz1[iv];
    }
}

END {
    process()
    
    file_version()
    header()
    format()

    structure()
    attributes()
}
