#!/usr/bin/sh

setup() {
    make -C ../tools/rbc install    
}

pre() {
    XS=40
    YS=52
    ZS=20
    df=1.0

    D="-XS=$XS -YS=$YS -ZS=$ZS"

    radius=3
    fraction=0.5
    sc=1 ang=0
    plcmt.ro $XS $YS $ZS $radius $fraction $sc $ang ic_solid.txt rbcs-ic.txt

    rm -rf diag.txt h5 bop ply solid-ply
    cp sdf/gx/small.rot.dat sdf.dat
    cp data/sphere_R1.ply mesh_solid.ply

    argp .conf.gx.base.h $D                \
         -rbcs -solids                       \
         -tend=3000.0 -part_freq=1000        \
         -walls -wall_creation=1000          \
         -pushflow -driving_force=$df        \
         -field_dumps -part_dumps -field_freq=1000 > .conf.h
    
}

compile() {
    { make clean && make -j ; } > /dev/null
}

. ./gx.panda.sh
#. ./gx.daint.sh

setup
pre
ini
compile
run
