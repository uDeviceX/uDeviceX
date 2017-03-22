#!/bin/bash

# Run from this directory:
#  > atest run_diag.sh
#
# To update the test change TEST to cTEST and run
#  > atest run_diag.sh
# add crap from test_data/* to git

#### Double poiseuille
# TEST: double.poiseuille
# set -x
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# argp .conf.double.poiseuille.h     \
#    -rsph=4 -pin_com=true           \
#   -tend=2.0 -steps_per_dump=4000    \
#   -pushtheflow -doublepoiseuille   \
#   -hdf5field_dumps -hdf5part_dumps \
#   -steps_per_hdf5dump=4000 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# ./udx
# avg_h52.m h5/flowfields-0001.h5 | fround.awk -v tol=1 > h5.out.txt

####
# TEST: diag.t4
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# cp sdf/wall1/wall.dat sdf.dat
# argp .conf.test.h  \
#    -rsph=4 -pin_com=true                                         \
#   -tend=1.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300 > .conf.h
# make clean && make -j && make -C ../tools
# ./udx
# avg_h5.m h5/flowfields-0003.h5 | fround.awk -v tol=1 > h5.out.txt

####
# TEST: diag.t5
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# cp sdf/wall1/wall.dat sdf.dat
# argp .conf.poiseuille.h \
#    -rsph=4 -pin_com=true                                         \
#   -tend=2.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# ./udx
# avg_h5.m h5/flowfields-0013.h5 | fround.awk -v tol=2 > h5.out.txt

####
# TEST: diag.t6
# export PATH=../tools:$PATH
# rm -rf ply h5 diag.txt
# cp sdf/cyl1/cyl.dat sdf.dat
# argp .conf.poiseuille.h \
#    -rsph=4 -pin_com=true                                         \
#   -tend=4.0 -steps_per_dump=300 -walls -wall_creation_stepid=100 \
#   -hdf5field_dumps -hdf5part_dumps -steps_per_hdf5dump=300 -pushtheflow > .conf.h
# make clean && make -j && make -C ../tools
# ./udx
# mid_h5.m h5/flowfields-0026.h5 | fhash.awk -v tol=2 > h5.out.txt


