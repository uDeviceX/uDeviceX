#!/usr/bin/sh

export PATH=../tools:$PATH

XS=40
YS=52
ZS=20

plcmt.ro

rm -rf diag.txt h5 bop
cp sdf/gx/small.rot.dat sdf.dat


argp .conf.gx.h                 \
   -rbcs                        \
   -tend=3000.0 -part_freq=5000 \
   -walls -wall_creation=1000   \
   -pushflow                   \
   -field_dumps -part_dumps -field_freq=5000 > .conf.h

{ make clean && make -j ; } > /dev/null
./udx
avg_h52.m h5/flowfields-0010.xmf | uscale 0.1 > h5.out.txt
