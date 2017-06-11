#!/usr/bin/sh

# Sergey's current run

. $HOME/.udx/u.sh
. ./gx.generic.sh

inc ./gx.HOST.sh

pre() {
    clean
    nv=498
    sdf=$googlex/8/150.sdf # set in gx.*.sh
    rbc=cells/sph.$nv.off
    sld=data/cylinder.ply
    copy $sdf $rbc $sld

    NX=2  NY=2  NZ=1
    XS=40 YS=52 ZS=20
    geom # set NN, LX, LY, LZ, Domain

    df=10.0
    fraction=0.2 radius=2.3 sc=0.2 ang=rnd
    plcmt.ro $LX $LY $LZ $radius $fraction $sc $ang ic_solid.txt rbcs-ic.txt

    argp .conf.gx.base.h $Domain              \
	 RBCnv=$nv                            \
	 rbcs solids contactforces            \
	 tend=3000.0 part_freq=100            \
	 walls wall_creation=1                \
	 pushflow     driving_force=$df       \
	 field_dumps part_dumps field_freq=100 > .conf.h
}

setup
pre
ini
compile
run
