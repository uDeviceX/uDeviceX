#!/bin/bash

set -eu

# load utilities
. u/`u.host`

n=40
d=/scratch/snx3000/lisergey/u
Time=00:25:00

s=../../src
u.conf $s u/safe base.h <<EOF
       DBG_PEEK
       ODSTR_SAFE
       run
EOF
u.make -j > make.log

safe_cp "$d/$n/sdf.dat" .

x=$n y=1 z=1
u.strtdir  . $x $y $z
u.batch   $x $y $z ./udx $Time
