#!/usr/bin/sh

export PATH=../tools:$PATH

XS=40
YS=52
ZS=20


D="-XS=$XS -YS=$YS -ZS=$ZS"

radius=5
fraction=0.1
sc=1 ang=0
plcmt.ro $XS $YS $ZS $radius $fraction $sc $ang ic_solid.txt rbcs-ic.txt

rm -rf diag.txt h5 bop ply
cp sdf/gx/small.rot.dat sdf.dat


argp .conf.gx.base.h $D                \
   -rbcs                               \
   -tend=3000.0 -part_freq=5000        \
   -walls -wall_creation=1000          \
   -pushflow                           \
   -field_dumps -part_dumps -field_freq=5000 > .conf.h

{ make clean && make -j ; } > /dev/null
./udx
