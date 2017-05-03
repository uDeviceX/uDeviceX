#!/usr/bin/sh

for f in $@
do
    bop2vtk "${f%.bop}.vtk" $f
done
