#!/usr/bin/sh

# assumes executed from the git repo;
# if one need to dump elsewhere, use symlinks
. ../generic.sh #defines SRC via GITROOT
$SRC/configure $SRC

. ./ic.sh

pre() {
    clean

    DATA=$SRC/data
    sdf=$DATA/sdf/yplates1/yplates.dat
    rig=$DATA/rig/sphere.ply

    cp $sdf sdf.dat
    cp $rig mesh_solid.ply

    NX=1  NY=1  NZ=1
    XS=64 YS=32 ZS=16
    geom # set NN, LX, LY, LZ, Domain
    ic_center
    Tend=100
    gamma_dot=0.01
    
    u.conf $SRC conf.base.h <<EOF
    $Domain tend=$Tend
    gamma_dot=$gamma_dot shear_y
    field_dumps part_dumps
    field_freq=4000 parts_freq=1000
    solids sbounce_back
    walls wall_creation=5000
    strt_dumps strt_freq=20000
    run
EOF
    u.strtdir . $NX $NY $NZ
}

run() {
    sh runfile
}

setup
pre
compile
run
