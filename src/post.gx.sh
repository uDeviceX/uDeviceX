#!/bin/bash

build_ply mesh_solid.ply solid_diag*.txt

PLY=solid-ply
mkdir -p $PLY
mv solid*.ply $PLY

../tools/allbop2vtk.sh bop/*.bop
