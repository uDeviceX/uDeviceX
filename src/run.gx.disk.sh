#!/usr/bin/sh

setup() {
    make -C ../tools/rbc install
    make -C ../tools install
    make -C ../post/build_smesh install
}

pre() {

    NX=1  NY=1  NZ=1
    #NX=2  NY=2  NZ=1
    NN=$((${NX}*${NY}*${NZ}))
    
    XS=40 YS=52 ZS=20
    LX=$((${NX}*${XS}))
    LY=$((${NY}*${YS}))
    LZ=$((${NZ}*${ZS}))

    df=0.1

    D="-XS=$XS -YS=$YS -ZS=$ZS"

    radius=2.3
    fraction=0.2
    sc=0.2 ang=0
    plcmt.ro $LX $LY $LZ $radius $fraction $sc $ang ic_solid.txt rbcs-ic.txt

    rm -rf diag.txt h5 bop ply solid-ply solid_diag*txt
    cp sdf/gx/small.rot.dat sdf.dat
    #cp ~/geoms/128.dat sdf.dat
    cp data/cylinder.ply mesh_solid.ply
    cp cells/sph.498.off  rbc.off

    #ply.sxyz xs ys zs in.ply > out.ply
    
    argp .conf.gx.base.h $D                  \
         -rbcs -solids -contactforces        \
         -tend=3000.0 -part_freq=500        \
         -walls -wall_creation=1          \
         -pushflow -driving_force=$df        \
         -field_dumps -part_dumps -field_freq=500 > .conf.h
}

compile() {
    { make clean && make -j ; } > /dev/null
}

#. ./gx.panda.sh
. ./gx.daint.sh

setup
pre
ini
compile
run
