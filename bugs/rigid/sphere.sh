#!/bin/bash

# assumes executed from the git repo;
# if one need to dump elsewhere, use symlinks
. ../../run/generic.sh #defines SRC via GITROOT
$SRC/configure $SRC

. ./ic.sh

pre() {
    clean

    DATA=$SRC/data
    sdf=$DATA/sdf/yplates1/yplates.dat
    rig=$DATA/rig/sphere.ply

    cp $sdf sdf.dat
    cp $rig mesh_solid.ply

    NX=2  NY=1  NZ=1
    XS=32 YS=32 ZS=32
    geom # set NN, LX, LY, LZ, Domain
    ic_center
    
    u.conf $SRC conf.base.h <<EOF
    $Domain
    run
EOF
    u.strtdir . $NX $NY $NZ
}

run() {
    u.run $NX $NY $NZ ./udx
}

setup
pre
compile
run
