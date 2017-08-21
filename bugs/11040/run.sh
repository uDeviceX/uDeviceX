#!/bin/bash

set -eu

# load utilities
. u/`u.host`

n=40
d=/scratch/snx3000/lisergey/u # data

s=../../src
echo run | u.conf $s u/x base.h
u.make -j > make.log

safe_cp "$d/$n/sdf.dat" .

x=$n y=1 z=1
u.strtdir  . $x $y $z
u.batch   $x $y $z ./udx
