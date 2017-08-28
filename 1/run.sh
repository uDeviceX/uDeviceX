#!/bin/bash

set -eu

# load utilities
. u/`u.host`

n=1
d=/scratch/snx3000/lisergey/u
Time=00:25:00
u=u/safe
s=../src

u.conf $s $u base.h <<EOF
       DBG_PEEK
       ODSTR_SAFE
       run
EOF
u.make -j > make.log

safe_cp "$d/40/sdf.dat" .

x=$n y=1 z=1
u.strtdir  . $x $y $z
u.batch   $x $y $z ./udx $Time
