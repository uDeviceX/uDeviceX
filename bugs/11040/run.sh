#!/bin/bash

set -eu

n=40
d=/scratch/snx3000/lisergey/u # data

s=../../src
echo run | u.conf $s u/x base.h
u.make -j

cp $n/sdf.dat .

x=$n y=1 z=1
u.strtdir  . $x $y $z
u.run    $x $y $z ./udx 
