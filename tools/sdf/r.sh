#!/bin/bash

f=../../src/sdf/gx/small.dat
r=../../src/sdf/gx/small.rot.dat # rotated

b=../../src/sdf/gx/small.rot.bov # bov

printf 'r.sh: writing: %s %s\n' $r $b > /dev/stderr

sdf.shuffle yxz $f $r
sdf.2bov $r $b
