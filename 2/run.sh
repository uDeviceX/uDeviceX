#!/bin/bash

set -eu

# load utilities
. u/`u.host`
. u/common

n=2
d=/scratch/snx3000/lisergey/u
Time=10:00:00
u=u/dang
s=../src

(cd $s/../cmd; make)
u.conf $s $u base.h <<EOF
       run
EOF
u.make -j > make.log

x=$n y=1 z=1

u.strtdir . $x $y $z
u.batch     $x $y $z ./udx $Time
