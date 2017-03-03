#!/bin/bash

# A wrapper for `iotags' ("in-out" tags)

setenv () {
    export PATH=../../../post/vrbc:$PATH
    export nb=498
    export xl=0 yl=0 zl=0 xh=192 yh=32 zh=192
    export pbcx=1 pbcy=1 pbcz=1
    export nvar=9 # number of variables in *.data files ([rva][xyz] (9
		  # in total))
}

data ()  { # get Athena's data
    d=/scratch/snx3000/eceva/2_tcflow/tc2_192x32x192_Ht15_cons/uDevX_omega_0.05
    if test -f "$sol"; then return; fi

    scp daint:"$d"/src/ply/rbcs-1000.ply .
    scp daint:"$d"/src/dpd-data/"$sol" .
    scp daint:"$d"/src/rbc-data/"$rbc" .
}

build () {
    (cd ../../../post/vrbc && make)
    make
}

faces=test_data/faces.bin # binary file with a structure of one RBC
sol=Bdpd_data-1000.data
rbc=Brbc_data-1000.data
data
build
setenv
./iotags "$faces" "$rbc" "$sol"    tags.bin
tags2vtk "$sol"   tags.bin           in.vtk
