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

    x0=`echo "$LX * 0.5" | bc`
    y0=`echo "$LY * 0.5" | bc`
    z0=`echo "$LZ * 0.5" | bc` 
    echo $x0 $y0 $z0 > ic_solid.txt

    Tend=100
    gamma_dot=0.01
    
    u.conf run/rigid/conf.h <<EOF
    $Domain tend=$Tend
    gamma_dot=$gamma_dot
    field_dumps part_dumps
    field_freq=4000 parts_freq=1000
    solids sbounce_back
    walls wall_creation=5000
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
