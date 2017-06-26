#!/usr/bin/sh

# assume the script is ran from uDeviceX/src/ directory

./configure
. ./run/generic.sh

pre() {
    clean

    sdf=data/sdf/yplates1/yplates.dat
    rig=data/rig/sphere.ply

    cp $sdf sdf.dat
    cp $rig mesh_solid.ply

    NX=1  NY=1  NZ=1
    XS=32 YS=32 ZS=16
    geom # set NN, LX, LY, LZ, Domain
    
    Tend=100
    gamma_dot=0.01
    
    u.conf run/rigid/conf.h <<EOF
    $Domain tend=$Tend
    gamma_dot=$gamma_dot
    field_dumps part_dumps
    field_freq=4000 parts_freq=1000
    solids sbounce_back
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
