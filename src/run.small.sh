#!/usr/bin/sh

. $HOME/.udx/u.sh # u.sh should be installed; make install -C ../tools
inc ./gx.HOST.sh

setup() {
    make -C ../tools/rbc install
    make -C ../tools install
    make -C ../post/build_smesh install
}

pre() {
    nv=162
    NX=1  NY=1  NZ=1
    #NX=2  NY=2  NZ=1
    NN=$((${NX}*${NY}*${NZ}))

    XS=20 YS=24 ZS=12
    LX=$((${NX}*${XS}))
    LY=$((${NY}*${YS}))
    LZ=$((${NZ}*${ZS}))

    df=0
    D="-XS=$XS -YS=$YS -ZS=$ZS"

    radius=2.0
    fraction=0.2
    sc=1.0 ang=0
    plcmt.ro $LX $LY $LZ $radius $fraction $sc $ang ic_solid.txt rbcs-ic.txt

    rm -rf diag.txt h5 bop ply solid-ply solid_diag*txt
    cp sdf/gx/small.rot.dat sdf.dat
    #cp ~/geoms/128.dat sdf.dat
    cp data/cylinder.ply mesh_solid.ply
    cp cells/sph.$nv.off  rbc.off

    #ply.sxyz xs ys zs in.ply > out.ply

    argp .conf.gx.base.h $D                  \
	 -RBCnv=$nv                          \
	 -rbcs -solids -contactforces        \
	 -tend=3000.0 -part_freq=500        \
	 -walls -wall_creation=1          \
	 -pushflow -driving_force=$df        \
	 -field_dumps -part_dumps -field_freq=500 > conf.h
}

compile() {
    { make clean && make -j ; } > /dev/null
}

setup
pre
ini
compile
run
